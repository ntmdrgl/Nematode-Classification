import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from evaluate_cam import save_classifier, temporal_predictions
from models import create_model
from models.dilated_fcn import DilatedFCN


class DilatedFCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_paper_convolution_sequence(self):
        model = DilatedFCN()
        convs = [layer for layer in model.features.modules() if isinstance(layer, nn.Conv2d)]
        self.assertEqual([layer.out_channels for layer in convs], [16, 32, 64, 128, 256, 512, 512, 512, 512, 512])
        self.assertEqual([layer.stride for layer in convs], [(1, 1), (2, 2), (2, 2), (1, 2), (1, 2), (1, 2)] + [(1, 1)] * 4)
        self.assertEqual([layer.dilation for layer in convs[-4:]], [(1, 2), (1, 4), (1, 2), (1, 1)])
        self.assertEqual(convs[0].kernel_size, (7, 7))
        self.assertTrue(all(layer.kernel_size == (3, 3) for layer in convs[1:]))
        self.assertEqual(model.classifier.kernel_size, (1, 1))

    def test_variable_counts_and_temporal_resolution(self):
        for channels, length in [(1, 1), (2, 33), (6, 257), (17, 96)]:
            with self.subTest(channels=channels, length=length), torch.inference_mode():
                model = DilatedFCN(in_channels=channels, num_classes=3).eval()
                inputs = torch.randn(2, channels, length)
                features = model.features(inputs)
                self.assertEqual(features.shape, (2, 512, (channels + 3) // 4, (length + 31) // 32))
                probabilities, recording_logits = temporal_predictions(model, inputs)
                self.assertEqual(probabilities.shape, (2, length, 3))
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, length))
                self.assertTrue(torch.isfinite(probabilities).all())
                torch.testing.assert_close(recording_logits, model(inputs))
                # Native (uninterpolated) CAM logits must reconstruct GAP logits.
                native_logits = model.classifier(features).mean(dim=(2, 3))
                torch.testing.assert_close(native_logits, recording_logits, atol=1e-6, rtol=1e-5)

    def test_training_gradients_and_checkpoint_roundtrip(self):
        model = DilatedFCN(in_channels=1, num_classes=3)
        inputs = torch.randn(2, 1, 65)
        loss = nn.CrossEntropyLoss()(model(inputs), torch.tensor([0, 2]))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.features[1][0].weight.grad.abs().sum().item(), 0)
        model.eval()
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path = Path(directory) / "checkpoint.pt"
            torch.save(model.state_dict(), path)
            restored = DilatedFCN(in_channels=1, num_classes=3).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(inputs), model(inputs))

    def test_factory_and_classifier_export(self):
        params = SimpleNamespace(in_channels=6, num_classes=3)
        for name, num_features in [("fcn", 128), ("dilated_fcn", 512)]:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                model = create_model(name, params).eval()
                output = Path(directory)
                save_classifier(model, output, ["a", "b", "c"])
                saved = torch.load(output / "classifier_weights.pt", weights_only=True)
                torch.testing.assert_close(saved["state_dict"]["weight"], model.classifier.weight)
                with (output / "classifier_weights.csv").open(newline="") as file:
                    rows = list(csv.reader(file))
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(len(row) == num_features + 2 for row in rows))

    def test_original_fcn_cam_is_unchanged(self):
        model = create_model("fcn", SimpleNamespace(in_channels=6, num_classes=5)).eval()
        inputs = torch.randn(2, 6, 65)
        with torch.inference_mode():
            expected = model.classifier(model.features(inputs).transpose(1, 2)).softmax(-1)
            probabilities, logits = temporal_predictions(model, inputs)
            torch.testing.assert_close(probabilities, expected)
            torch.testing.assert_close(logits, model(inputs))


if __name__ == "__main__":
    unittest.main()
