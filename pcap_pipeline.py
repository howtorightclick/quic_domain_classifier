"""Convert PCAP/PCAPNG to classifier JSONL using Wireshark's TShark dissector."""
import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


V1 = 0x00000001
V2 = 0x6B3343CF


class CaptureError(ValueError):
    """Capture cannot be represented faithfully in the classifier schema."""


def fields(node, name):
    return node.findall(f".//field[@name='{name}']")


def number(node, name, default=None):
    matches = fields(node, name)
    if not matches:
        if default is not None:
            return default
        raise CaptureError(f'Missing dissector field: {name}')
    # PDML show is the decoded integer; value can contain masked wire bytes.
    value = matches[0].get('show', '')
    try:
        return int(value, 16 if value.lower().startswith('0x') else 10)
    except ValueError as exc:
        raise CaptureError(f'Invalid numeric dissector field: {name}') from exc


def initial_frames(node):
    unavailable = ('quic.decryption_failed', 'quic.ft.unknown',
                   'quic.remaining_payload', '_ws.malformed')
    if any(fields(node, name) for name in unavailable):
        return None
    frame_nodes = fields(node, 'quic.frame')
    if not frame_nodes:
        # Absence of a frame tree does not prove successful decryption.
        return None
    result = []
    for frame in frame_nodes:
        kind = number(frame, 'quic.frame_type')
        if kind == 0:
            result.append({'type': 'PADDING', 'length': number(frame, 'quic.padding_length')})
        elif kind == 6:
            result.append({'type': 'CRYPTO', 'length': number(frame, 'quic.crypto.length')})
        else:
            result.append({'type': {1: 'PING', 2: 'ACK', 3: 'ACK'}.get(kind, f'0x{kind:x}')})
    return result


def quic_packet(node):
    length = number(node, 'quic.packet_length')
    if length <= 0:
        raise CaptureError('Invalid QUIC packet length')
    if number(node, 'quic.header_form') == 0:
        return {'type': '1-RTT', 'length': length}
    version = number(node, 'quic.version')
    if version == 0:
        kind = 'VN'
    elif version == V1:
        kind = ('Initial', '0-RTT', 'Handshake', 'Retry')[
            checked_type(number(node, 'quic.long.packet_type'))]
    elif version == V2:
        kind = ('Retry', 'Initial', '0-RTT', 'Handshake')[
            checked_type(number(node, 'quic.long.packet_type_v2'))]
    else:
        raise CaptureError(f'Unsupported QUIC version: 0x{version:08x}')
    result = {'type': kind, 'length': length}
    if kind == 'Initial':
        scid = number(node, 'quic.scil')
        if not 0 <= scid <= 20:
            raise CaptureError('Invalid SCID length')
        result.update(scid_length=scid, initial_frames=initial_frames(node))
    return result


def checked_type(value):
    if value not in range(4):
        raise CaptureError('Invalid long-header packet type')
    return value


def endpoints(node):
    # Refuse tunnels/multiple network layers rather than mix inner and outer endpoints.
    ips = node.findall("./proto[@name='ip']") + node.findall("./proto[@name='ipv6']")
    udps = node.findall("./proto[@name='udp']")
    if len(ips) != 1 or len(udps) != 1:
        raise CaptureError('Expected one IP layer and one UDP datagram')
    prefix = ips[0].get('name')
    def address(side):
        matches = fields(ips[0], f'{prefix}.{side}')
        if len(matches) != 1:
            raise CaptureError('Missing or ambiguous IP endpoint')
        return str(ipaddress.ip_address(matches[0].get('show')))
    return ((address('src'), number(udps[0], 'udp.srcport')),
            (address('dst'), number(udps[0], 'udp.dstport')))


def pdml_packets(stream):
    """Release each XML packet after consumption instead of loading the capture XML."""
    context = ET.iterparse(stream, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag == 'packet':
            yield elem
            root.remove(elem)


def records_from_pdml(stream, capture_id, group_id, domain=None, client_addresses=(),
                      max_datagrams=512, skip_invalid=False, expected_flow=None):
    if max_datagrams < 1:
        raise ValueError('max_datagrams must be positive')
    clients = {str(ipaddress.ip_address(a)) for a in client_addresses}
    connections, rejected = {}, set()
    report = Counter()
    for node in pdml_packets(stream):
        quics = node.findall("./proto[@name='quic']")
        if not quics:
            continue
        report['quic_datagrams'] += 1
        cid = None
        frame_id = number(node, 'frame.number', -1)
        try:
            src, dst = endpoints(node)
            if expected_flow is not None and (src, dst) not in (expected_flow, expected_flow[::-1]):
                report['datagrams_outside_session_flow'] += 1
                continue
            ids = {number(f, 'quic.connection.number') for f in quics
                   if fields(f, 'quic.connection.number')}
            cid = next(iter(ids)) if len(ids) == 1 else None
            if cid in rejected:
                continue
            if cid is None:
                raise CaptureError('Missing or ambiguous QUIC connection number')
            if fields(node, '_ws.malformed') or node.findall(".//proto[@name='_ws.malformed']"):
                raise CaptureError('Malformed packet reported by TShark')
            if number(node, 'frame.cap_len') < number(node, 'frame.len'):
                raise CaptureError('Truncated capture frame')
            if fields(node, 'quic.stateless_reset'):
                raise CaptureError('Stateless reset is not a supported packet type')
            padding = 0
            packet_nodes = quics
            if fields(quics[-1], 'quic.coalesced_padding_data'):
                tail = quics[-1]
                if fields(tail, 'quic.header_form') or len(quics) < 2:
                    raise CaptureError('Ambiguous appended datagram padding')
                padding = int(tail.get('size', '0'))
                if padding <= 0:
                    raise CaptureError('Invalid appended datagram padding length')
                packet_nodes = quics[:-1]
            packets = [quic_packet(q) for q in packet_nodes]
            length = number(node, 'udp.length') - 8
            if length <= 0 or sum(p['length'] for p in packets) + padding != length:
                raise CaptureError('QUIC packet lengths do not account for the UDP payload')
            if any(p['type'] in ('Retry', 'VN') for p in packets) and len(packets) != 1:
                raise CaptureError('Retry/VN must be standalone')
            if any(p['type'] == '1-RTT' for p in packets[:-1]):
                raise CaptureError('Short header must be last')
            state = connections.get(cid)
            if state is None:
                if clients:
                    if (src[0] in clients) == (dst[0] in clients):
                        raise CaptureError('Exactly one endpoint must match --client-address')
                    client, server = (src, dst) if src[0] in clients else (dst, src)
                else:
                    # A decrypted ClientHello proves the role, unlike port heuristics.
                    hello = any(f.get('show') == '1' for f in fields(node, 'tls.handshake.type'))
                    if packets[0]['type'] != 'Initial' or not hello:
                        raise CaptureError('Client role unavailable; supply --client-address or capture the ClientHello')
                    client, server = src, dst
                record = {'connection_id': f'{capture_id}:quic:{cid}', 'group_id': group_id,
                          'datagrams': []}
                if domain is not None:
                    record['domain'] = domain
                state = connections[cid] = {'client': client, 'server': server, 'record': record}
            if src == state['client'] and dst == state['server']:
                direction = 'c2s'
            elif src == state['server'] and dst == state['client']:
                direction = 's2c'
            else:
                raise CaptureError('Endpoint migration is not supported by this extractor')
            # Keep ciphertext-derived frame measurements only for server Initials.
            if direction == 'c2s':
                for packet in packets:
                    packet.pop('initial_frames', None)
            ds = state['record']['datagrams']
            if len(ds) < max_datagrams:
                datagram = {'direction': direction, 'length': length, 'packets': packets}
                if padding:
                    datagram['udp_padding_length'] = padding
                ds.append(datagram)
            else:
                report['datagrams_beyond_window'] += 1
        except (CaptureError, ValueError) as exc:
            if not skip_invalid:
                raise CaptureError(f'Frame {frame_id}, connection {cid}: {exc}') from exc
            report[f'rejected: {exc}'] += 1
            if cid is not None:
                rejected.add(cid)
                connections.pop(cid, None)
    records = [s['record'] for s in connections.values()]
    report['connections_written'] = len(records)
    report['connections_rejected'] = len(rejected)
    return records, dict(report)


def find_tshark(executable=None):
    found = shutil.which(executable or 'tshark')
    if not found and executable is None:
        candidate = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Wireshark/tshark.exe'
        if candidate.is_file():
            found = str(candidate)
    if not found:
        raise CaptureError('TShark not found. Install Wireshark with TShark, or pass --tshark PATH.')
    return found


def extract_capture(path, tshark, **kwargs):
    # The Windows TShark build can reject paths beyond MAX_PATH even when
    # Python can read them. Stage only these inputs under a short temporary path.
    if os.name == 'nt' and len(str(Path(path).resolve())) >= 260:
        with tempfile.TemporaryDirectory(prefix='quic-') as directory:
            staged = Path(directory) / 'capture.pcapng'
            shutil.copyfile(path, staged)
            return extract_capture(staged, tshark, **kwargs)
    ports = kwargs.pop('ports', ())
    if not ports and kwargs.get('expected_flow'):
        # Receipt-backed sessions are QUIC; a client ephemeral port can otherwise
        # select an unrelated registered UDP dissector before QUIC detection.
        ports = [kwargs['expected_flow'][1][1]]
    command = [tshark, '-n', '-2', '-r', str(path), '-Y', 'quic', '-T', 'pdml']
    for port in ports:
        command.extend(['-d', f'udp.port=={port},quic'])
    # A file for stderr prevents a full pipe from deadlocking a large capture.
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
            try:
                result = records_from_pdml(process.stdout, **kwargs)
            except ET.ParseError as exc:
                process.kill()
                process.wait()
                errors.seek(0)
                diagnostic = errors.read().decode('utf-8', errors='replace').strip()
                raise CaptureError(f'Invalid TShark PDML: {diagnostic or str(exc)}') from exc
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                process.stdout.close()
            code = process.wait()
        errors.seek(0)
        diagnostic = errors.read().decode('utf-8', errors='replace').strip()
        if code:
            raise CaptureError(f'TShark exited with {code}: {diagnostic}')
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture_inputs(paths):
    """Expand verified defense/domain/session folders, excluding diagnostic captures."""
    inputs = []
    for path in paths:
        if not path.is_dir():
            inputs.append((path, {}))
            continue
        sessions = sorted(p for p in (path / 'data' / 'verified').glob('*/*/*') if p.is_dir())
        if not sessions:
            raise CaptureError(f'No verified session folders found in {path}')
        for session in sessions:
            receipt_path = session / 'receipt.json'
            direct = [session / 'pcaps' / name for name in ('traffic.pcap', 'traffic.pcapng')
                      if (session / 'pcaps' / name).is_file()]
            if len(direct) > 1:
                raise CaptureError(f'Ambiguous canonical captures in {session}')
            if not receipt_path.is_file():
                if not direct:
                    raise CaptureError(f'No canonical traffic capture or receipt in {session}')
                domain = session.parent.name
                # Conservatively keep repeated domain/session IDs together across parts/defenses.
                group = 'folder-session:' + hashlib.sha256(
                    json.dumps([domain, session.name]).encode()).hexdigest()
                inputs.append((direct[0], {'domain': domain, 'group_id': group}))
                continue
            try:
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                mode, domain = receipt_path.parts[-4:-2]
                if (receipt['verified'] is not True or receipt['hostname'] != domain or
                        receipt['mode'] != mode):
                    raise ValueError('Receipt verification or folder labels do not match')
                relative = PurePosixPath(receipt['artifact_paths']['pcap'])
                if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
                    raise ValueError('Invalid canonical capture path')
                base = (receipt_path.parent / 'pcaps').resolve()
                capture = (direct[0] if direct else base.joinpath(*relative.parts)).resolve()
                if not capture.is_relative_to(base) or not capture.is_file():
                    raise ValueError('Canonical capture is missing or outside the session')
                if capture.name not in ('traffic.pcap', 'traffic.pcapng'):
                    raise ValueError('Expected a canonical traffic capture')
                client, client_port, server, server_port, protocol = receipt['five_tuple']
                if protocol != 17 or any(type(p) is not int or not 1 <= p <= 65535
                                         for p in (client_port, server_port)):
                    raise ValueError('Expected a UDP five-tuple with valid ports')
                client, server = str(ipaddress.ip_address(client)), str(ipaddress.ip_address(server))
                study, slot = receipt['study_sha256'], receipt['slot']
                if not isinstance(study, str) or not study or type(slot) is not int or slot < 0:
                    raise ValueError('Missing study or session slot')
                # Keep matching study/domain slots together across traffic modes.
                group = 'session:' + hashlib.sha256(
                    json.dumps([study, domain, slot]).encode()).hexdigest()
                inputs.append((capture, {'domain': domain, 'group_id': group,
                                         'client_addresses': [client],
                                         'expected_flow': ((client, client_port), (server, server_port))}))
            except (KeyError, TypeError, ValueError) as exc:
                raise CaptureError(f'Invalid receipt {receipt_path}: {exc}') from exc
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='+', type=Path,
                        help='PCAP files or part directories containing data/verified/defense/domain/id sessions')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--domain', help='Ground-truth label applied to ALL emitted connections')
    parser.add_argument('--group-id', help='Collection unit shared by these captures; defaults to capture SHA256')
    parser.add_argument('--client-address', action='append', default=[], help='Client IP; repeat for multiple clients')
    parser.add_argument('--port', action='append', type=int, default=[], help='Force QUIC decode on this UDP port')
    parser.add_argument('--tshark', help='Path to TShark executable (Wireshark 4.2+ recommended)')
    parser.add_argument('--max-datagrams', type=int, default=512)
    parser.add_argument('--skip-invalid', action='store_true', help='Discard entire identifiable invalid connections')
    args = parser.parse_args()
    if args.max_datagrams < 1 or any(not 1 <= p <= 65535 for p in args.port):
        parser.error('max-datagrams must be positive and ports must be 1..65535')
    if args.group_id == '' or args.domain == '':
        parser.error('group-id and domain must be nonempty when supplied')
    temporary = None
    try:
        inputs = capture_inputs(args.captures)
        if args.domain and any(meta and meta['domain'] != args.domain for _, meta in inputs):
            raise CaptureError('--domain conflicts with session domain labels')
        if args.output.resolve() in {p.resolve() for p, _ in inputs}:
            raise CaptureError('Output must not overwrite an input capture')
        tshark = find_tshark(args.tshark)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        seen, total = set(), 0
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent,
                                         suffix='.jsonl.tmp', delete=False) as output:
            temporary = Path(output.name)
            for capture, metadata in inputs:
                digest = file_digest(capture)
                if digest in seen:
                    raise CaptureError(f'Duplicate capture contents: {capture}')
                seen.add(digest)
                options = dict(metadata)
                options.update(group_id=args.group_id or metadata.get('group_id', digest),
                               domain=args.domain or metadata.get('domain'),
                               client_addresses=args.client_address or metadata.get('client_addresses', []))
                records, report = extract_capture(
                    capture, tshark, capture_id=digest, **options,
                    max_datagrams=args.max_datagrams, skip_invalid=args.skip_invalid, ports=args.port)
                from features import validate
                for record in records:
                    validate(record)
                    output.write(json.dumps(record, allow_nan=False) + '\n')
                total += len(records)
                print(json.dumps({'capture': str(capture), **report}))
        if not total:
            raise CaptureError('No usable QUIC connections found; check capture coverage and --port')
        os.replace(temporary, args.output)
        temporary = None
        print(json.dumps({'output': str(args.output), 'connections_written': total}))
    except (OSError, ValueError, ET.ParseError) as exc:
        parser.exit(1, f'Extraction failed: {exc}\n')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
