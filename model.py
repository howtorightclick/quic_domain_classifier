import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class AttentiveBiGRU(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True, bidirectional=True)
        self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, sequences):
        lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
        packed = pack_padded_sequence(pad_sequence(sequences, batch_first=True), lengths,
                                      batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)
        mask = torch.arange(encoded.size(1), device=encoded.device)[None] < lengths.to(encoded.device)[:, None]
        weights = self.score(encoded).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        return (encoded * weights.unsqueeze(-1)).sum(1), weights

class DomainClassifier(nn.Module):
    def __init__(self, packet_dim, burst_dim, session_dim, classes, hidden=64, dropout=.2):
        super().__init__()
        self.config = dict(packet_dim=packet_dim, burst_dim=burst_dim, session_dim=session_dim,
                           classes=classes, hidden=hidden, dropout=dropout)
        self.packet = nn.Sequential(nn.Linear(packet_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.burst = AttentiveBiGRU(hidden, hidden)
        self.session = AttentiveBiGRU(hidden * 2 + burst_dim, hidden)
        self.quic = nn.Sequential(nn.Linear(session_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, rows, return_embedding=False):
        device = next(self.parameters()).device
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        packets = [self.packet(tensor(p)) for r in rows for p in r['packets']]
        burst_vectors, _ = self.burst(packets)
        sessions, start = [], 0
        for r in rows:
            n = len(r['packets'])
            sessions.append(torch.cat([burst_vectors[start:start+n], tensor(r['bursts'])], -1))
            start += n
        session_vectors, _ = self.session(sessions)
        static = self.quic(torch.stack([tensor(r['session']) for r in rows]))
        embedding = torch.cat([session_vectors, static], -1)
        return embedding if return_embedding else self.head(embedding)
