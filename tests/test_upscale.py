from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

import krea2_studio.upscale as upscale_module


class UpscaleTests(unittest.TestCase):
    def tearDown(self) -> None:
        upscale_module.clear_upscaler_cache()

    def test_missing_model_uses_lanczos_and_reports_reason(self) -> None:
        source = Image.new("RGB", (10, 8), (20, 40, 60))
        result, metadata = upscale_module.upscale(
            source,
            scale=1.5,
            model_path=Path("definitely-missing-model.pth"),
            device="cpu",
        )

        self.assertEqual(result.size, (15, 12))
        self.assertEqual(metadata["backend"], "pillow_lanczos")
        self.assertIn("not found", metadata["reason"])

    def test_neural_result_is_resized_to_requested_scale_and_preserves_alpha(self) -> None:
        class FakeDescriptor:
            scale = 4
            dtype = torch.float32
            device = torch.device("cpu")

            def to(self, device: str):
                self.device = torch.device(device)
                return self

            def eval(self):
                return self

            def __call__(self, value):
                return torch.nn.functional.interpolate(value, scale_factor=4, mode="nearest")

        class FakeLoader:
            def load_from_file(self, path):
                return FakeDescriptor()

        fake_spandrel = types.SimpleNamespace(ImageModelDescriptor=FakeDescriptor, ModelLoader=FakeLoader)
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "4x-Test.pth"
            model.touch()
            source = Image.new("RGBA", (8, 6), (80, 120, 160, 100))
            with patch.dict(sys.modules, {"spandrel": fake_spandrel}):
                result, metadata = upscale_module.upscale(source, scale=2, model_path=model, device="cpu")

        self.assertEqual(result.size, (16, 12))
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getchannel("A").getextrema(), (100, 100))
        self.assertEqual(metadata["backend"], "spandrel")
        self.assertEqual(metadata["model"], "4x-Test.pth")
        self.assertEqual(metadata["native_scale"], 4)
        self.assertEqual(metadata["requested_scale"], 2)
        self.assertEqual(metadata["actual_scale"], 2)

    def test_incompatible_model_falls_back_without_hiding_failure(self) -> None:
        class FakeDescriptor:
            pass

        class WrongDescriptor:
            pass

        class FakeLoader:
            def load_from_file(self, path):
                return WrongDescriptor()

        fake_spandrel = types.SimpleNamespace(ImageModelDescriptor=FakeDescriptor, ModelLoader=FakeLoader)
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "not-an-image-model.pth"
            model.touch()
            with patch.dict(sys.modules, {"spandrel": fake_spandrel}):
                result, metadata = upscale_module.upscale(
                    Image.new("RGB", (4, 4)), scale=2, model_path=model, device="cpu"
                )

        self.assertEqual(result.size, (8, 8))
        self.assertEqual(metadata["backend"], "pillow_lanczos")
        self.assertIn("Expected an image-to-image model", metadata["reason"])

    def test_rejects_downscaling_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1.0"):
            upscale_module.upscale(Image.new("RGB", (4, 4)), scale=0.5, model_path=None, device="cpu")


if __name__ == "__main__":
    unittest.main()
