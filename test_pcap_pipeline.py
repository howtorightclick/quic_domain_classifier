"""Synthetic PDML tests; no TShark installation needed for these unit tests."""
import io
import json
import os
from pathlib import Path
import tempfile
import struct
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from pcap_pipeline import (CaptureError, V1, V2, main, quic_packet,
                           records_from_pdml, extract_capture, find_tshark)


def field(name, value):
    return f'<field name="{name}" show="{value}"/>'


def frame(kind, extra=''):
    return '<field name="quic.frame">' + field('quic.frame_type', kind) + extra + '</field>'


def quic(kind=0, length=1200, version=V1, frames='', connection=0, hello=False):
    body = field('quic.packet_length', length)
    if connection is not None:
        body += field('quic.connection.number', connection)
    body += field('quic.header_form', int(kind != 'short'))
    if kind != 'short':
        body += field('quic.version', hex(version)) + field('quic.scil', 8)
        body += field('quic.long.packet_type_v2' if version == V2 else 'quic.long.packet_type', kind)
    if hello:
        body += field('tls.handshake.type', 1)
    return '<proto name="quic">' + body + frames + '</proto>'


def packet(quics, size=1200, server=False, index=1, truncated=False, ipv6=False):
    client, remote = ('2001:db8::1', '2001:db8::2') if ipv6 else ('192.0.2.1', '192.0.2.2')
    src, dst = (remote, client) if server else (client, remote)
    prefix = 'ipv6' if ipv6 else 'ip'
    return ('<packet><proto name="frame">' + field('frame.number', index) +
            field('frame.len', size + 42) + field('frame.cap_len', size + (30 if truncated else 42)) +
            f'</proto><proto name="{prefix}">' + field(prefix + '.src', src) +
            field(prefix + '.dst', dst) + '</proto><proto name="udp">' +
            field('udp.srcport', 443 if server else 50000) +
            field('udp.dstport', 50000 if server else 443) + field('udp.length', size + 8) +
            '</proto>' + quics + '</packet>')


def parse(*packets, **kwargs):
    return records_from_pdml(io.BytesIO(('<pdml>' + ''.join(packets) + '</pdml>').encode()),
                             'capture-hash', 'collection', **kwargs)


class PipelineTest(unittest.TestCase):
    def test_receipt_flow_forces_quic_decode_unless_ports_overridden(self):
        flow = (('192.0.2.1', 41170), ('192.0.2.2', 443))
        for ports, expected in (([], 'udp.port==443,quic'), ([8443], 'udp.port==8443,quic')):
            with self.subTest(ports=ports), patch('pcap_pipeline.subprocess.Popen') as popen:
                process = popen.return_value.__enter__.return_value
                process.stdout = io.BytesIO(b'<pdml></pdml>')
                process.wait.return_value = 0
                extract_capture('capture.pcapng', 'tshark', capture_id='test', group_id='test',
                                expected_flow=flow, ports=ports)
                command = popen.call_args.args[0]
                self.assertEqual(command[-2:], ['-d', expected])

    def test_appended_udp_padding_preserves_packet_and_datagram_sizes(self):
        from features import extract, validate
        padding = '<proto name="quic" size="373">' + field('quic.coalesced_padding_data', '') + '</proto>'
        rows, _ = parse(packet(quic(length=827, hello=True) + padding))
        datagram = rows[0]['datagrams'][0]
        self.assertEqual(datagram['length'], 1200)
        self.assertEqual(datagram['packets'][0]['length'], 827)
        self.assertEqual(datagram['udp_padding_length'], 373)
        self.assertEqual(len(datagram['packets']), 1)
        self.assertEqual(extract(rows[0])['packets'][0][0, 0], 1200)
        for bad in (-1, 372, 373.5, True):
            datagram['udp_padding_length'] = bad
            with self.assertRaises(ValueError):
                validate(rows[0])
        with self.assertRaises(CaptureError):
            parse(packet(quic(length=827, hello=True) + padding.replace('373', '372')))

    def test_coalescing_frames_and_feature_compatibility(self):
        frames = (frame(6, field('quic.crypto.length', 100)) +
                  frame(0, field('quic.padding_length', 200)) + frame(3) + frame(1))
        rows, report = parse(packet(quic(hello=True)), packet(
            quic(length=400, frames=frames) + quic(2, 800, connection=None), server=True),
            packet(quic('short', 900), size=900, server=True),
            packet(quic('short', 80), size=80), domain='example.org')
        row = rows[0]
        self.assertEqual(report['connections_written'], 1)
        self.assertEqual(row['domain'], 'example.org')
        self.assertEqual([d['direction'] for d in row['datagrams']], ['c2s', 's2c', 's2c', 'c2s'])
        self.assertEqual(row['datagrams'][1]['packets'][0]['initial_frames'][2], {'type': 'ACK'})
        from features import extract
        features = extract(row)
        self.assertEqual(features['packets'][1].shape, (2, 38))
        self.assertEqual(features['bursts'].shape, (3, 18))
        self.assertEqual(features['session'].shape, (36,))
        self.assertAlmostEqual(features['packets'][1][0, 33], .5)
        self.assertAlmostEqual(features['session'][-1], 900 / 980, places=6)
        serialized = json.dumps(row)
        for forbidden in ('192.0.2', 'tls.', 'quic.scid', 'quic.dcid'):
            self.assertNotIn(forbidden, serialized)

    def test_v2_types_and_negotiation(self):
        for kind, expected in enumerate(('Retry', 'Initial', '0-RTT', 'Handshake')):
            self.assertEqual(quic_packet(ET.fromstring(quic(kind, version=V2)))['type'], expected)
        self.assertEqual(quic_packet(ET.fromstring(quic(version=0)))['type'], 'VN')
        with self.assertRaisesRegex(CaptureError, 'Unsupported'):
            quic_packet(ET.fromstring(quic(version=123)))

    def test_missing_and_failed_decryption(self):
        for frames in ('', frame(1) + field('quic.decryption_failed', '')):
            rows, _ = parse(packet(quic(hello=True)), packet(quic(frames=frames), server=True))
            self.assertIsNone(rows[0]['datagrams'][1]['packets'][0]['initial_frames'])

    def test_ambiguous_role_requires_explicit_client(self):
        with self.assertRaisesRegex(CaptureError, 'Client role unavailable'):
            parse(packet(quic(), server=True))
        rows, _ = parse(packet(quic(), server=True), client_addresses=['192.0.2.1'])
        self.assertEqual(rows[0]['datagrams'][0]['direction'], 's2c')

    def test_ipv6_and_short_header(self):
        rows, _ = parse(packet(quic('short'), ipv6=True), client_addresses=['2001:db8::1'])
        self.assertEqual(rows[0]['datagrams'][0]['packets'][0]['type'], '1-RTT')

    def test_bad_lengths_truncation_and_missing_connection(self):
        for p in (packet(quic(hello=True), size=1300),
                  packet(quic(hello=True), truncated=True),
                  packet(quic(hello=True, connection=None))):
            with self.assertRaises(CaptureError):
                parse(p)

    def test_skip_invalid_removes_entire_connection(self):
        rows, report = parse(packet(quic(hello=True)), packet(quic(), truncated=True),
                             packet(quic()), packet(quic(hello=True, connection=1)), skip_invalid=True)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['connection_id'].endswith(':1'))
        self.assertEqual(report['connections_rejected'], 1)

    def test_packet_order_constraints(self):
        for q in (quic('short', 600, hello=True) + quic(length=600),
                  quic(3, 600) + quic(length=600)):
            with self.assertRaises(CaptureError):
                parse(packet(q), client_addresses=['192.0.2.1'])

    def test_window_and_capture_order(self):
        rows, report = parse(packet(quic(hello=True)), packet(quic(), server=True), max_datagrams=1)
        self.assertEqual(len(rows[0]['datagrams']), 1)
        self.assertEqual(report['datagrams_beyond_window'], 1)

    def test_migration_rejected(self):
        migrated = packet(quic()).replace('192.0.2.1', '192.0.2.3')
        with self.assertRaisesRegex(CaptureError, 'migration'):
            parse(packet(quic(hello=True)), migrated)

    def test_output_unchanged_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, output = Path(directory) / 'a.pcap', Path(directory) / 'out.jsonl'
            capture.write_bytes(b'capture')
            output.write_text('existing', encoding='utf-8')
            with patch('sys.argv', ['pcap_pipeline.py', str(capture), '--output', str(output)]), \
                    patch('pcap_pipeline.find_tshark', return_value='tshark'), \
                    patch('pcap_pipeline.extract_capture', side_effect=CaptureError('bad capture')):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(output.read_text(encoding='utf-8'), 'existing')
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['a.pcap', 'out.jsonl'])

    def test_part_directory_cli(self):
        from pcap_pipeline import capture_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / 'data/verified/buflo/example.org/attempt-1'
            relative = 'corpus/example.org/buflo/session-000001/traffic.pcapng'
            capture = session / 'pcaps' / relative
            capture.parent.mkdir(parents=True)
            capture.write_bytes(b'canonical')
            diagnostics = capture.parent / 'diagnostics'
            diagnostics.mkdir()
            (diagnostics / 'debug.pcapng').write_bytes(b'debug')
            receipt = {'verified': True, 'hostname': 'example.org', 'mode': 'buflo',
                       'artifact_paths': {'pcap': relative}, 'study_sha256': 'study', 'slot': 1,
                       'five_tuple': ['192.0.2.1', 50000, '192.0.2.2', 443, 17]}
            receipt_path = session / 'receipt.json'
            receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
            inputs = capture_inputs([root])
            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0][0], capture.resolve())
            self.assertEqual(inputs[0][1]['domain'], 'example.org')
            output = root / 'out.jsonl'
            records, report = parse(packet(quic(hello=True)))
            with patch('sys.argv', ['pcap_pipeline.py', str(root), '--output', str(output)]), \
                    patch('pcap_pipeline.find_tshark', return_value='tshark'), \
                    patch('pcap_pipeline.extract_capture', return_value=(records, report)) as extract_mock, \
                    patch('sys.stdout', new_callable=io.StringIO):
                main()
            self.assertEqual(extract_mock.call_args.kwargs['domain'], 'example.org')
            self.assertEqual(extract_mock.call_args.kwargs['client_addresses'], ['192.0.2.1'])
            self.assertTrue(output.is_file())
            for bad in ('../../outside.pcapng', '/absolute.pcapng', 'missing.pcapng'):
                receipt['artifact_paths']['pcap'] = bad
                receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
                with self.assertRaises(CaptureError):
                    capture_inputs([root])
            receipt['artifact_paths']['pcap'] = relative
            receipt['hostname'] = 'wrong.org'
            receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
            with self.assertRaises(CaptureError):
                capture_inputs([root])

    def test_session_flow_excludes_unrelated_traffic(self):
        # Obtain the fixture's ports rather than assume its client port.
        from pcap_pipeline import endpoints
        flow = endpoints(ET.fromstring(packet(quic(hello=True))))
        unrelated = packet(quic(hello=True, connection=1)).replace('192.0.2.1', '192.0.2.3')
        rows, report = parse(packet(quic(hello=True)), unrelated,
                             packet(quic(), server=True), expected_flow=flow)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['datagrams']), 2)
        self.assertEqual(report['datagrams_outside_session_flow'], 1)

    def test_empty_part_rejected(self):
        from pcap_pipeline import capture_inputs
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CaptureError, 'No verified'):
                capture_inputs([Path(directory)])

    def test_direct_layout_with_arbitrary_defenses(self):
        from pcap_pipeline import capture_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(8):
                folder = root / 'data/verified' / f'custom-defense-{i}' / 'example.org/id-1/pcaps'
                folder.mkdir(parents=True)
                (folder / 'traffic.pcapng').write_bytes(str(i).encode())
                (folder / 'diagnostics').mkdir()
                (folder / 'diagnostics/debug.pcapng').write_bytes(b'debug')
            inputs = capture_inputs([root])
            self.assertEqual(len(inputs), 8)
            self.assertEqual({m['domain'] for _, m in inputs}, {'example.org'})
            self.assertEqual(len({m['group_id'] for _, m in inputs}), 1)
            self.assertTrue(all(p.name == 'traffic.pcapng' for p, _ in inputs))
            (folder / 'traffic.pcap').write_bytes(b'ambiguous')
            with self.assertRaisesRegex(CaptureError, 'Ambiguous'):
                capture_inputs([root])

    def test_cli_writes_compatible_jsonl(self):
        records, report = parse(packet(quic(hello=True)))
        with tempfile.TemporaryDirectory() as directory:
            capture, output = Path(directory) / 'a.pcap', Path(directory) / 'out.jsonl'
            capture.write_bytes(b'capture')
            with patch('sys.argv', ['pcap_pipeline.py', str(capture), '--output', str(output)]), \
                    patch('pcap_pipeline.find_tshark', return_value='tshark'), \
                    patch('pcap_pipeline.extract_capture', return_value=(records, report)), \
                    patch('sys.stdout', new_callable=io.StringIO):
                main()
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')), records[0])

    def test_real_pcap_with_tshark_when_installed(self):
        try:
            tshark = find_tshark()
        except CaptureError:
            self.skipTest('TShark not installed; real PCAP integration not exercised')
        # Valid v1 long-header boundaries with deliberately undecryptable ciphertext.
        payload = bytes.fromhex('c00000000108') + b'12345678' + b'\x08abcdefgh\x00'
        payload += struct.pack('!H', 0x4000 | 1174) + bytes(1174)
        udp = struct.pack('!HHHH', 50000, 443, len(payload) + 8, 0) + payload
        ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20 + len(udp), 1, 0, 64, 17, 0,
                         bytes([192, 0, 2, 1]), bytes([192, 0, 2, 2]))
        total = sum(struct.unpack('!10H', ip))
        total = (total & 65535) + (total >> 16)
        checksum = (~((total & 65535) + (total >> 16))) & 65535
        ip = ip[:10] + struct.pack('!H', checksum) + ip[12:]
        ethernet = bytes(12) + b'\x08\x00' + ip + udp
        pcap = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
        pcap += struct.pack('<IIII', 1, 0, len(ethernet), len(ethernet)) + ethernet
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.pcap'
            path.write_bytes(pcap)
            records, _ = extract_capture(path, tshark, capture_id='test', group_id='test',
                                         client_addresses=['192.0.2.1'], ports=[443])
            if os.name == 'nt':
                long_path = Path(directory) / ('a' * 110) / ('b' * 110) / 'synthetic.pcap'
                long_path.parent.mkdir(parents=True)
                long_path.write_bytes(pcap)
                self.assertGreaterEqual(len(str(long_path)), 260)
                long_records, _ = extract_capture(long_path, tshark, capture_id='test', group_id='test',
                                                  client_addresses=['192.0.2.1'], ports=[443])
                self.assertEqual(long_records, records)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['datagrams'][0]['length'], 1200)
        self.assertEqual(records[0]['datagrams'][0]['packets'][0]['type'], 'Initial')


if __name__ == '__main__':
    unittest.main()
