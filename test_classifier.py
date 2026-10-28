import copy
import unittest
import numpy as np
from features import extract, Normalizer


def fixture():
    return {'connection_id': 'example-1', 'group_id': 'capture-1', 'domain': 'example.org', 'datagrams': [
        {'direction': 'c2s', 'length': 1200, 'packets': [{'type': 'Initial', 'length': 1200}]},
        {'direction': 's2c', 'length': 1200, 'packets': [
            {'type': 'Initial', 'length': 400, 'scid_length': 8, 'initial_frames': [
                {'type': 'CRYPTO', 'length': 100}, {'type': 'PADDING', 'length': 200}, {'type': 'ACK'}]},
            {'type': 'Handshake', 'length': 800}]},
        {'direction': 's2c', 'length': 900, 'packets': [{'type': '1-RTT', 'length': 900}]},
        {'direction': 'c2s', 'length': 80, 'packets': [{'type': '1-RTT', 'length': 80}]}]}


class FeaturesTest(unittest.TestCase):
    def test_bursts_lengths_and_masks(self):
        row = extract(fixture())
        self.assertEqual([len(b) for b in row['packets']], [1, 2, 1])
        self.assertEqual(row['bursts'][1, 1], 2100)
        self.assertAlmostEqual(row['session'][-1], 900 / 980, places=6)
        self.assertAlmostEqual(row['packets'][1][0, 33], .5)
        self.assertTrue(np.isnan(row['packets'][0][0, -1]))
        transformed = Normalizer().fit([row]).transform(row)
        self.assertTrue(np.isfinite(transformed['session']).all())
        self.assertTrue(all(np.isfinite(p).all() for p in transformed['packets']))
        self.assertEqual(transformed['packets'][0][0, -1], 0)
        self.assertEqual(transformed['packets'][1][0, -1], 1)

    def test_prefix_consistency(self):
        a, b = fixture(), fixture()
        b['datagrams'][-1]['length'] = 999
        b['datagrams'][-1]['packets'][0]['length'] = 999
        np.testing.assert_equal(extract(a, 2)['session'], extract(b, 2)['session'])

    def test_missing_frames_differ_from_observed_empty(self):
        a, b = fixture(), fixture()
        a['datagrams'][1]['packets'][0]['initial_frames'] = None
        b['datagrams'][1]['packets'][0]['initial_frames'] = []
        self.assertTrue(np.isnan(extract(a)['packets'][1][0, -1]))
        self.assertEqual(extract(b)['packets'][1][0, -1], 0)

    def test_invalid_data(self):
        bad = fixture()
        bad['datagrams'][0]['length'] = 2
        with self.assertRaises(ValueError):
            extract(bad)
        with self.assertRaises(ValueError):
            extract({'datagrams': []})


try:
    import torch
    from model import DomainClassifier
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class MetricsTest(unittest.TestCase):
    def test_imbalanced_classification(self):
        from run import metrics
        report = metrics([0, 0, 0, 1], [0, 0, 1, 1], ['a', 'b'])
        self.assertEqual(report['confusion_matrix'], [[2, 1], [0, 1]])
        self.assertEqual(report['sample_count'], 4)
        self.assertEqual(report['accuracy'], .75)
        self.assertAlmostEqual(report['balanced_accuracy'], 5 / 6)
        self.assertAlmostEqual(report['macro_precision'], .75)
        self.assertAlmostEqual(report['macro_recall'], 5 / 6)
        self.assertAlmostEqual(report['macro_f1'], 11 / 15)
        self.assertAlmostEqual(report['weighted_precision'], .875)
        self.assertAlmostEqual(report['weighted_recall'], .75)
        self.assertAlmostEqual(report['weighted_f1'], 23 / 30)
        self.assertEqual(report['per_domain']['a']['support'], 3)

    def test_missing_and_unpredicted_classes(self):
        from run import metrics
        import json
        report = metrics([0, 0, 1], [0, 0, 0], ['a', 'b', 'c'])
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertAlmostEqual(report['macro_recall'], 1 / 3)
        self.assertEqual(report['per_domain']['b']['f1'], 0)
        self.assertEqual(report['per_domain']['c'],
                         {'precision': 0., 'recall': 0., 'f1': 0., 'support': 0})
        json.dumps(report, allow_nan=False)

    def test_perfect_predictions(self):
        from run import metrics
        report = metrics([0, 1], [0, 1], ['a', 'b'])
        for key in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall',
                    'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1'):
            self.assertEqual(report[key], 1., key)

    def test_invalid_metric_inputs(self):
        from run import metrics
        for truth, pred in (([], []), ([0, 1], [0]), ([-1], [0]),
                            ([0], [2]), ([0.5], [0]), ([[0]], [[0]])):
            with self.subTest(truth=truth, pred=pred), self.assertRaises(ValueError):
                metrics(truth, pred, ['a', 'b'])
        for labels in ([], ['a', 'a']):
            with self.assertRaises(ValueError):
                metrics([0], [0], labels)


@unittest.skipIf(torch is None, 'PyTorch not installed')
class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        a, b = fixture(), fixture()
        b['datagrams'] = b['datagrams'][:1]
        rows = [extract(a), extract(b)]
        norm = Normalizer().fit(rows)
        self.rows = [norm.transform(r) for r in rows]
        r = self.rows[0]
        self.model = DomainClassifier(r['packets'][0].shape[1], r['bursts'].shape[1], len(r['session']), 2, hidden=8, dropout=0)

    def test_padding_does_not_change_predictions(self):
        self.model.eval()
        with torch.no_grad():
            batch = self.model(self.rows)
            singles = torch.cat([self.model([r]) for r in self.rows])
        torch.testing.assert_close(batch, singles, atol=1e-6, rtol=1e-5)

    def test_gradients_reach_all_branches_and_loss_falls(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.02)
        target = torch.tensor([0, 1])
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.model(self.rows), target)
            loss.backward()
            for name in ('packet', 'burst', 'session', 'quic', 'head'):
                grads = [p.grad for p in getattr(self.model, name).parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] / 2)

    def test_group_leakage_rejected(self):
        from run import check_splits
        a, b = fixture(), copy.deepcopy(fixture())
        b['connection_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'group_id'):
            check_splits([a], [b])

if __name__ == '__main__':
    unittest.main()
