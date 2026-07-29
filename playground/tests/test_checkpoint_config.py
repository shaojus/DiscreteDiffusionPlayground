import unittest

from omegaconf import OmegaConf

from playground.eval import resolve_checkpoint_config


def _runtime_config(include_encoding=True):
    data = {
        "dataset": "gmm",
        "n_bits": 8,
        "n_mixes": 8,
        "interleave": False,
        "reverse": False,
    }
    if include_encoding:
        data["encoding"] = "gray"
    return OmegaConf.create(
        {
            "seed": 91,
            "model": {"_target_": "runtime.Model", "d_model": 16},
            "data": data,
        }
    )


class CheckpointConfigurationTests(unittest.TestCase):
    def test_saved_model_data_and_seed_are_authoritative(self):
        checkpoint = {
            "cfg": {
                "seed": 24,
                "model": {"_target_": "saved.Model", "d_model": 32},
                "data": {
                    "dataset": "checkerboard",
                    "n_bits": 6,
                    "encoding": "gray",
                    "interleave": True,
                    "reverse": True,
                },
            }
        }
        model_cfg, data_cfg, seed = resolve_checkpoint_config(
            checkpoint,
            _runtime_config(),
        )
        self.assertEqual(model_cfg._target_, "saved.Model")
        self.assertEqual(data_cfg.dataset, "checkerboard")
        self.assertEqual(data_cfg.n_bits, 6)
        self.assertEqual(data_cfg.encoding, "gray")
        self.assertTrue(data_cfg.interleave)
        self.assertTrue(data_cfg.reverse)
        self.assertEqual(seed, 24)

    def test_saved_data_missing_encoding_defaults_to_binary(self):
        checkpoint = {
            "cfg": {
                "model": {"_target_": "saved.Model"},
                "data": {"dataset": "gmm", "n_bits": 4, "n_mixes": 2},
            }
        }
        _, data_cfg, _ = resolve_checkpoint_config(checkpoint, _runtime_config())
        self.assertEqual(data_cfg.encoding, "binary")

    def test_missing_saved_data_falls_back_to_runtime_data(self):
        checkpoint = {"cfg": {"model": {"_target_": "saved.Model"}}}
        _, data_cfg, seed = resolve_checkpoint_config(checkpoint, _runtime_config())
        self.assertEqual(data_cfg.dataset, "gmm")
        self.assertEqual(data_cfg.encoding, "gray")
        self.assertEqual(seed, 91)

    def test_missing_checkpoint_config_falls_back_to_runtime(self):
        model_cfg, data_cfg, seed = resolve_checkpoint_config(
            {},
            _runtime_config(),
        )
        self.assertEqual(model_cfg._target_, "runtime.Model")
        self.assertEqual(data_cfg.dataset, "gmm")
        self.assertEqual(data_cfg.encoding, "gray")
        self.assertEqual(seed, 91)

    def test_saved_model_hidden_dimension_is_resolved_for_instantiation(self):
        checkpoint = {
            "cfg": {
                "model": {
                    "_target_": "saved.Model",
                    "d_model": 32,
                    "d_hid": None,
                    "d_hid_factor": 4,
                },
                "data": {"dataset": "gmm", "n_bits": 4},
            }
        }
        model_cfg, _, _ = resolve_checkpoint_config(
            checkpoint,
            _runtime_config(),
        )
        self.assertEqual(model_cfg.d_hid, 128)
        self.assertNotIn("d_hid_factor", model_cfg)

    def test_missing_encoding_everywhere_defaults_to_binary(self):
        _, data_cfg, _ = resolve_checkpoint_config(
            {},
            _runtime_config(include_encoding=False),
        )
        self.assertEqual(data_cfg.encoding, "binary")


if __name__ == "__main__":
    unittest.main()
