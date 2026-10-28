"""Features from parsed QUIC metadata; no payload, hostname, IP or CID values."""
import math
import numpy as np

TYPES = ('Initial', 'Handshake', '0-RTT', '1-RTT', 'Retry', 'VN')
STATS = ('count', 'mean', 'std', 'min', 'p25', 'median', 'p75', 'max')

def stats(values):
    if not values:
        return [0.] * 8
    a = np.asarray(values, dtype=float)
    return [len(a), a.mean(), a.std(), a.min(), *np.percentile(a, [25, 50, 75]), a.max()]

def presence(ds):
    found = {p['type'] for d in ds for p in d['packets']}
    return [float(t in found) for t in TYPES]

def lengths(ds, kind):
    return [p['length'] for d in ds for p in d['packets'] if p['type'] == kind]

def coalescing(ds):
    return sum(len(d['packets']) > 1 for d in ds) / len(ds) if ds else 0.

def packet_features(d):
    server = d['direction'] == 's2c'
    initial = [p for p in d['packets'] if server and p['type'] == 'Initial']
    decrypted = bool(initial) and all(p.get('initial_frames') is not None for p in initial)
    frames = [f for p in initial for f in p['initial_frames']] if decrypted else []
    crypto = [f['length'] for f in frames if f['type'] == 'CRYPTO']
    fvec = stats(crypto) + [sum(f.get('length', 0) for f in frames if f['type'] == 'PADDING') /
                           max(1, sum(p['length'] for p in initial))] + [
        float(any(f['type'] == t for f in frames)) for t in ('CRYPTO', 'ACK', 'PADDING', 'PING')]
    if not decrypted:
        fvec = [float('nan')] * len(fvec)
    return [d['length'], float(server), len(d['packets']), *presence([d]),
            *stats(lengths([d], 'Initial') if server else []),
            *stats(lengths([d], 'Handshake') if server else []), *fvec]

def burst_features(ds):
    return [len(ds), sum(d['length'] for d in ds), *stats([d['length'] for d in ds]),
            *presence(ds), coalescing(ds), float(ds[0]['direction'] == 's2c')]

def validate(record):
    if not record['datagrams']:
        raise ValueError('Connection must contain at least one datagram')
    for d in record['datagrams']:
        if d['direction'] not in ('c2s', 's2c') or not d['packets']:
            raise ValueError('Invalid direction or empty packet list')
        if not math.isfinite(d['length']) or d['length'] <= 0:
            raise ValueError('Datagram length must be finite positive UDP payload bytes')
        for p in d['packets']:
            if p['type'] not in TYPES or not math.isfinite(p['length']) or p['length'] <= 0:
                raise ValueError('Invalid QUIC packet type or length')
            for f in p.get('initial_frames') or []:
                if f['type'] in ('CRYPTO', 'PADDING') and (not math.isfinite(f['length']) or f['length'] < 0):
                    raise ValueError('CRYPTO/PADDING requires nonnegative length')
        padding = d.get('udp_padding_length', 0)
        if type(padding) is not int or padding < 0:
            raise ValueError('UDP padding length must be a nonnegative integer')
        if sum(p['length'] for p in d['packets']) + padding != d['length']:
            raise ValueError('QUIC packet lengths plus UDP padding must sum to UDP payload length')

def extract(record, max_datagrams=512):
    if max_datagrams < 1:
        raise ValueError('max_datagrams must be positive')
    validate(record)
    ds = record['datagrams'][:max_datagrams]
    bursts = []
    for d in ds:
        if not bursts or bursts[-1][-1]['direction'] != d['direction']:
            bursts.append([])
        bursts[-1].append(d)
    server = [d for d in ds if d['direction'] == 's2c']
    one = [d for d in ds if any(p['type'] == '1-RTT' for p in d['packets'])]
    sb = sum(d['length'] for d in one if d['direction'] == 's2c')
    cb = sum(d['length'] for d in one if d['direction'] == 'c2s')
    first_initial = next((p for d in server for p in d['packets'] if p['type'] == 'Initial'), {})
    scid = first_initial.get('scid_length')
    if scid is not None and (not isinstance(scid, int) or not 0 <= scid <= 20):
        raise ValueError('SCID length must be 0..20 bytes or null')
    session = [float('nan') if scid is None else scid,
               *stats(lengths(server, 'Initial')), *stats(lengths(server, 'Handshake')),
               sum(lengths(server, 'Handshake')), coalescing(server),
               *(presence(server[:1]) if server else [float('nan')] * 6),
               float(any(p['type'] == 'Retry' for d in ds for p in d['packets'])),
               float(any(p['type'] == 'VN' for d in ds for p in d['packets'])),
               *stats([d['length'] for d in one if d['direction'] == 's2c']),
               sb / (sb + cb) if sb + cb else float('nan')]
    return {'packets': [np.asarray([packet_features(d) for d in b], dtype=np.float32) for b in bursts],
            'bursts': np.asarray([burst_features(b) for b in bursts], dtype=np.float32),
            'session': np.asarray(session, dtype=np.float32)}

class Normalizer:
    """Training-only fit, signed log1p scaling, mean imputation, observed masks."""
    def fit(self, rows):
        self.state = {}
        for key in ('packets', 'bursts', 'session'):
            arrays = ([p for r in rows for p in r[key]] if key == 'packets' else
                      [r[key] for r in rows] if key == 'bursts' else [r[key][None] for r in rows])
            a = np.concatenate(arrays)
            a = np.sign(a) * np.log1p(np.abs(a))
            valid = np.isfinite(a)
            count = valid.sum(0).clip(1)
            mean = np.where(valid, a, 0).sum(0) / count
            variance = np.where(valid, (a - mean) ** 2, 0).sum(0) / count
            self.state[key] = {'mean': mean.tolist(), 'scale': np.sqrt(variance).clip(.1).tolist()}
        return self

    def transform(self, row):
        def scale(a, key):
            s = self.state[key]
            valid = np.isfinite(a)
            logged = np.sign(a) * np.log1p(np.abs(a))
            z = np.where(valid, (logged - s['mean']) / s['scale'], 0)
            return np.concatenate([z, valid.astype(float)], axis=-1).astype(np.float32)
        return {'packets': [scale(a, 'packets') for a in row['packets']],
                'bursts': scale(row['bursts'], 'bursts'), 'session': scale(row['session'], 'session')}
