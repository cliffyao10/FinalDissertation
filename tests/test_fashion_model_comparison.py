import unittest

import torch

from recommendation_training.fashion_model_comparison import (
    BiLSTMCompatibilityBaseline,
    SetTransformerCompatibilityBaseline,
    TypeAwarePairwiseBaseline,
    infer_matched_replacement_groups,
    paired_bootstrap_auc_comparisons,
    ranking_metrics,
)


class TensorOutfitDataset:
    def __init__(self, embeddings, masks, labels):
        self.embeddings = embeddings
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)


class SumScorer(torch.nn.Module):
    def forward(self, embeddings, mask):
        return (embeddings * mask.float().unsqueeze(-1)).sum(dim=(1, 2))


class FashionModelComparisonTests(unittest.TestCase):
    def test_ranking_metrics_marks_published_unpaired_rows_unavailable(self):
        class Dataset:
            labels = torch.tensor([1.0, 1.0, 0.0])
            masks = torch.ones(3, 4, dtype=torch.bool)
            embeddings = torch.zeros(3, 4, 8)

            def __len__(self):
                return len(self.labels)

        model = torch.nn.Sequential()
        report = ranking_metrics(model, Dataset(), "cpu")
        self.assertEqual(
            report["fitb4_protocol"], "unavailable_for_unpaired_published_rows"
        )
        self.assertIsNone(report["pair_ranking_accuracy"])

    def test_representative_models_return_one_logit_per_outfit(self):
        embeddings = torch.randn(3, 4, 8)
        masks = torch.tensor(
            [
                [True, True, True, True],
                [True, False, True, True],
                [False, True, True, True],
            ]
        )
        models = (
            BiLSTMCompatibilityBaseline(8, hidden_dim=4),
            TypeAwarePairwiseBaseline(8, pair_dim=3, hidden_dim=5),
            SetTransformerCompatibilityBaseline(
                8, model_dim=8, heads=2, layers=1, dropout=0.0
            ),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(tuple(model(embeddings, masks).shape), (3,))

    def test_pair_recovery_requires_one_changed_slot(self):
        positive = torch.ones(4, 2)
        negative = positive.clone()
        negative[2] = 0
        dataset = TensorOutfitDataset(
            torch.stack((positive, negative)),
            torch.ones(2, 4, dtype=torch.bool),
            torch.tensor([1.0, 0.0]),
        )
        groups = infer_matched_replacement_groups(dataset)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["slot"], 2)

    def test_ranking_metrics_construct_four_choice_diagnostic(self):
        examples = []
        masks = []
        labels = []
        for value in (4.0, 5.0, 6.0):
            positive = torch.ones(4, 1) * value
            negative = positive.clone()
            negative[1] = 0.0
            examples.extend((positive, negative))
            masks.extend((torch.ones(4, dtype=torch.bool),) * 2)
            labels.extend((1.0, 0.0))
        dataset = TensorOutfitDataset(
            torch.stack(examples), torch.stack(masks), torch.tensor(labels)
        )
        metrics = ranking_metrics(SumScorer(), dataset, "cpu", seed=9, batch_size=4)
        self.assertEqual(metrics["matched_pairs"], 3)
        self.assertEqual(metrics["fitb4_examples"], 3)
        self.assertEqual(metrics["pair_ranking_accuracy"], 1.0)
        self.assertEqual(metrics["fitb4_accuracy"], 1.0)

    def test_paired_bootstrap_reports_positive_auc_difference(self):
        labels = [0, 0, 1, 1]
        runs = {
            "baseline": [{"test_probabilities": [0.4, 0.6, 0.5, 0.7]}],
            "proposed": [{"test_probabilities": [0.1, 0.2, 0.8, 0.9]}],
        }
        result = paired_bootstrap_auc_comparisons(
            runs, labels, "proposed", iterations=50, seed=3
        )
        self.assertGreater(result["baseline"]["proposed_minus_baseline_auc"], 0)


if __name__ == "__main__":
    unittest.main()
