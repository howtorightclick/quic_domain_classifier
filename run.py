"""Train, evaluate and predict using JSONL records. Run --help for arguments."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from features import extract, Normalizer
from model import DomainClassifier


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('Empty dataset')
    return rows


def check_splits(train, valid):
    for key in ('connection_id', 'group_id'):
        for rows in (train, valid):
            if any(not isinstance(r.get(key), str) or not r[key] for r in rows):
                raise ValueError(f'Every labeled record needs a nonempty {key}')
        if {r[key] for r in train} & {r[key] for r in valid}:
            raise ValueError(f'Training and validation overlap on {key}')
    for rows in (train, valid):
        if len({r['connection_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate connection IDs in split')
    labels = sorted({r['domain'] for r in train})
    if len(labels) < 2:
        raise ValueError('Training requires at least two domains')
    if set(labels) != {r['domain'] for r in valid}:
        raise ValueError('Validation must contain every training domain and no unknown domains')
    return labels


def metrics(truth, pred, labels):
    """Closed-set scores; undefined per-class scores are zero.

    Macro scores include all checkpoint labels. Balanced accuracy averages
    recall only over classes present in the evaluated ground truth.
    """
    truth, pred = np.asarray(truth), np.asarray(pred)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Metrics require nonempty, unique labels')
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape or not truth.size:
        raise ValueError('Metrics require nonempty, equally sized 1-D targets and predictions')
    for values in (truth, pred):
        if (not np.issubdtype(values.dtype, np.integer) or
                np.any(values < 0) or np.any(values >= len(labels))):
            raise ValueError('Targets and predictions must be integer label indices in range')
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for y, p in zip(truth, pred):
        cm[y, p] += 1
    tp = cm.diagonal()
    precision = tp / cm.sum(0).clip(1)
    recall = tp / cm.sum(1).clip(1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = cm.sum(1)
    report = {'accuracy': float(tp.sum() / cm.sum()),
            'balanced_accuracy': float(recall[support > 0].mean()),
            'sample_count': int(cm.sum()),
            'per_domain': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i])}
                           for i, name in enumerate(labels)}, 'confusion_matrix': cm.tolist(), 'labels': labels}
    for name, values in (('precision', precision), ('recall', recall), ('f1', f1)):
        report[f'macro_{name}'] = float(values.mean())
        report[f'weighted_{name}'] = float(np.average(values, weights=support))
    return report


@torch.no_grad()
def probabilities(model, rows, batch_size):
    model.eval()
    return torch.cat([model(rows[i:i+batch_size]).softmax(-1).cpu()
                      for i in range(0, len(rows), batch_size)]).numpy()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    raw_train, raw_valid = read(args.train), read(args.valid)
    labels = check_splits(raw_train, raw_valid)
    train_rows = [extract(r, args.max_datagrams) for r in raw_train]
    normalizer = Normalizer().fit(train_rows)
    train_rows = [normalizer.transform(r) for r in train_rows]
    valid_rows = [normalizer.transform(extract(r, args.max_datagrams)) for r in raw_valid]
    y = [labels.index(r['domain']) for r in raw_train]
    vy = [labels.index(r['domain']) for r in raw_valid]
    first = train_rows[0]
    model = DomainClassifier(first['packets'][0].shape[-1], first['bursts'].shape[-1],
                             len(first['session']), len(labels)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = np.bincount(y, minlength=len(labels))
    weights = torch.tensor(len(y) / (len(labels) * counts), dtype=torch.float32, device=args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    best, stale = -1., 0
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(y))
        total = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model([train_rows[i] for i in ids])
            loss = loss_fn(logits, torch.tensor([y[i] for i in ids], device=args.device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(ids)
        report = metrics(vy, probabilities(model, valid_rows, args.batch_size).argmax(1), labels)
        print(json.dumps({'epoch': epoch + 1, 'loss': total / len(y),
                          'validation_macro_f1': report['macro_f1'],
                          'validation_metrics': report}))
        if report['macro_f1'] > best:
            best, stale = report['macro_f1'], 0
            torch.save({'schema_version': 1, 'state_dict': model.state_dict(), 'config': model.config,
                        'normalizer': normalizer.state, 'labels': labels, 'max_datagrams': args.max_datagrams,
                        'train_groups': sorted({r['group_id'] for r in raw_train}),
                        'valid_groups': sorted({r['group_id'] for r in raw_valid}),
                        'train_connections': sorted({r['connection_id'] for r in raw_train}),
                        'valid_connections': sorted({r['connection_id'] for r in raw_valid}),
                        'validation_metrics': report, 'seed': args.seed}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(json.dumps({'checkpoint': args.checkpoint, 'best_validation_macro_f1': best}))


def infer(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint['schema_version'] != 1:
        raise ValueError('Unsupported feature schema')
    model = DomainClassifier(**checkpoint['config']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    normalizer = Normalizer()
    normalizer.state = checkpoint['normalizer']
    raw = read(args.input)
    labels = checkpoint['labels']
    if args.command == 'evaluate':
        used_groups = set(checkpoint['train_groups'] + checkpoint['valid_groups'])
        used_ids = set(checkpoint['train_connections'] + checkpoint['valid_connections'])
        if any(not r.get('group_id') or not r.get('connection_id') or
               r['group_id'] in used_groups or r['connection_id'] in used_ids for r in raw):
            raise ValueError('Test data must have IDs and groups disjoint from train and validation')
        if len({r['connection_id'] for r in raw}) != len(raw):
            raise ValueError('Duplicate test connection IDs')
        if any(r.get('domain') not in labels for r in raw):
            raise ValueError('Closed-set evaluation requires known domain labels')
    rows = [normalizer.transform(extract(r, checkpoint['max_datagrams'])) for r in raw]
    probs = probabilities(model, rows, args.batch_size)
    if args.command == 'evaluate':
        print(json.dumps(metrics([labels.index(r['domain']) for r in raw], probs.argmax(1), labels), indent=2))
    else:
        for record, p in zip(raw, probs):
            print(json.dumps({'connection_id': record.get('connection_id'), 'domain': labels[int(p.argmax())],
                              'probabilities': dict(zip(labels, map(float, p)))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'predict', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--batch-size', type=int, default=16)
        p.add_argument('--device', default='cpu')
        if command == 'train':
            p.add_argument('--train', required=True)
            p.add_argument('--valid', required=True)
            p.add_argument('--epochs', type=int, default=50)
            p.add_argument('--patience', type=int, default=8)
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--max-datagrams', type=int, default=512)
        else:
            p.add_argument('--input', required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.command == 'train' and min(args.epochs, args.patience, args.max_datagrams) < 1):
        parser.error('Batch size, epochs, patience and maximum datagrams must be positive')
    train(args) if args.command == 'train' else infer(args)

if __name__ == '__main__':
    main()
