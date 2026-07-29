import unittest

import numpy as np
import torch

from playground.data import build_binary_2d_dataset
from playground.data.checkerboard import CheckerboardBinaryStream
from playground.data.encoding import binary_to_gray, gray_to_binary
from playground.data.gmm_binary_stream import GMMBinaryStream


class _FixedDistribution:
    def __init__(self, xy):
        self.xy = torch.as_tensor(xy, dtype=torch.float32)

    def sample(self):
        return self.xy.clone()


def _expected_sequence(values, n_bits, encoding, interleave, reverse):
    values = torch.as_tensor(values, dtype=torch.long)
    if encoding == "gray":
        values = binary_to_gray(values)
    shifts = torch.arange(n_bits - 1, -1, -1)
    bits = (values.unsqueeze(-1) >> shifts) & 1
    if interleave:
        sequence = torch.stack((bits[0], bits[1]), dim=1).reshape(-1)
    else:
        sequence = torch.cat((bits[0], bits[1]))
    if reverse:
        sequence = torch.flip(sequence, [0])
    return sequence


class EncodingUtilityTests(unittest.TestCase):
    def test_known_gray_values(self):
        values = torch.arange(8, dtype=torch.long)
        expected = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4])
        torch.testing.assert_close(binary_to_gray(values), expected)

    def test_gray_round_trip(self):
        for n_bits in range(1, 9):
            values = torch.arange(1 << n_bits, dtype=torch.long)
            decoded = gray_to_binary(binary_to_gray(values), n_bits)
            torch.testing.assert_close(decoded, values)


class DatasetEncodingTests(unittest.TestCase):
    dataset_classes = (CheckerboardBinaryStream, GMMBinaryStream)

    def _make_dataset(self, dataset_class, encoding, interleave, reverse):
        common = {
            "R": 1.0,
            "n_bits": 3,
            "encoding": encoding,
            "interleave": interleave,
            "reverse": reverse,
        }
        if dataset_class is CheckerboardBinaryStream:
            dataset = dataset_class(rotate_45=False, random_shift=False, **common)
            dataset.sample_xy = lambda: torch.tensor([-0.5, 0.5])
        else:
            dataset = dataset_class(
                n_mixes=1,
                realized_state={"loc": torch.zeros(1, 2)},
                **common,
            )
            dataset.dist = _FixedDistribution([-0.5, 0.5])
        return dataset

    def test_binary_sequence_is_unchanged(self):
        expected = torch.tensor([0, 1, 0, 1, 1, 0])
        for dataset_class in self.dataset_classes:
            with self.subTest(dataset=dataset_class.__name__):
                dataset = self._make_dataset(dataset_class, "binary", False, False)
                torch.testing.assert_close(next(iter(dataset)), expected)

    def test_all_encoding_and_order_combinations_round_trip(self):
        for dataset_class in self.dataset_classes:
            for encoding in ("binary", "gray"):
                for interleave in (False, True):
                    for reverse in (False, True):
                        with self.subTest(
                            dataset=dataset_class.__name__,
                            encoding=encoding,
                            interleave=interleave,
                            reverse=reverse,
                        ):
                            dataset = self._make_dataset(
                                dataset_class,
                                encoding,
                                interleave,
                                reverse,
                            )
                            sequence = next(iter(dataset))
                            expected = _expected_sequence(
                                [2, 6],
                                n_bits=3,
                                encoding=encoding,
                                interleave=interleave,
                                reverse=reverse,
                            )
                            torch.testing.assert_close(sequence, expected)
                            self.assertEqual(sequence.dtype, torch.long)
                            self.assertEqual(tuple(sequence.shape), (6,))
                            self.assertTrue(torch.all((sequence == 0) | (sequence == 1)))
                            np.testing.assert_allclose(
                                dataset.decode(sequence.numpy()),
                                np.array([-0.5, 0.5]),
                            )

    def test_invalid_encoding_is_rejected(self):
        for dataset_class in self.dataset_classes:
            with self.subTest(dataset=dataset_class.__name__):
                with self.assertRaises(ValueError):
                    dataset_class(encoding="not-an-encoding")


class DatasetFactoryTests(unittest.TestCase):
    def test_factory_defaults_legacy_config_to_binary(self):
        dataset = build_binary_2d_dataset({"dataset": "gmm", "n_bits": 4, "n_mixes": 2})
        self.assertEqual(dataset.encoding, "binary")
        self.assertFalse(dataset.interleave)
        self.assertFalse(dataset.reverse)

    def test_factory_flows_gray_and_order_settings(self):
        dataset = build_binary_2d_dataset(
            {
                "dataset": "checkerboard",
                "n_bits": 4,
                "encoding": "gray",
                "interleave": True,
                "reverse": True,
                "rotate_45": False,
            }
        )
        self.assertEqual(dataset.encoding, "gray")
        self.assertTrue(dataset.interleave)
        self.assertTrue(dataset.reverse)

    def test_realized_gmm_state_restores_locations(self):
        torch.manual_seed(10)
        original = build_binary_2d_dataset(
            {"dataset": "gmm", "n_bits": 4, "n_mixes": 3}
        )
        restored = build_binary_2d_dataset(
            {"dataset": "gmm", "n_bits": 4, "n_mixes": 3},
            realized_state=original.realized_state_dict(),
        )
        torch.testing.assert_close(restored.loc, original.loc)

    def test_legacy_seed_replay_matches_training_order(self):
        for config in (
            {"dataset": "checkerboard", "n_bits": 4},
            {"dataset": "gmm", "n_bits": 4, "n_mixes": 3},
        ):
            with self.subTest(dataset=config["dataset"]):
                torch.manual_seed(123)
                training_dataset = build_binary_2d_dataset(config)
                training_next_random = torch.rand(4)

                torch.manual_seed(123)
                evaluation_dataset = build_binary_2d_dataset(config)
                evaluation_next_random = torch.rand(4)

                if config["dataset"] == "gmm":
                    torch.testing.assert_close(
                        evaluation_dataset.loc,
                        training_dataset.loc,
                    )
                torch.testing.assert_close(evaluation_next_random, training_next_random)


if __name__ == "__main__":
    unittest.main()
