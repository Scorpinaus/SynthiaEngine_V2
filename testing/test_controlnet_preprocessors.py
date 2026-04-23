import asyncio
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image

from backend.adapters.controlnet_preprocessors import (
    AnylinePreprocessor,
    ControlNetAuxPreprocessor,
    NormalBaePreprocessor,
    SamMobilePreprocessor,
    TEEDPreprocessor,
    get_preprocessor,
    list_preprocessors,
)
from backend.main import list_controlnet_preprocessors, run_controlnet_preprocessor
from backend.workflow import _controlnet_preprocess


def _png_upload(name: str = "input.png") -> UploadFile:
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    payload = BytesIO()
    image.save(payload, format="PNG")
    payload.seek(0)
    return UploadFile(filename=name, file=payload)


class ControlNetPreprocessorValidationTests(unittest.TestCase):
    def test_canny_coerces_valid_values(self):
        preprocessor = get_preprocessor("canny")
        validated = preprocessor.validate_params(
            {"low_threshold": "32", "high_threshold": 128.0}
        )
        self.assertEqual(validated["low_threshold"], 32)
        self.assertEqual(validated["high_threshold"], 128)

    def test_canny_rejects_unknown_param(self):
        preprocessor = get_preprocessor("canny")
        with self.assertRaisesRegex(
            ValueError, "Unsupported params for preprocessor 'canny'"
        ):
            preprocessor.validate_params({"foo": 1})

    def test_canny_rejects_out_of_bounds(self):
        preprocessor = get_preprocessor("canny")
        with self.assertRaisesRegex(ValueError, "must be <= 255"):
            preprocessor.validate_params({"low_threshold": 0, "high_threshold": 999})

    def test_normal_bae_uses_base_controlnet_aux_run(self):
        self.assertIs(NormalBaePreprocessor.run, ControlNetAuxPreprocessor.run)

    def test_preprocessor_catalog_exposes_param_schema(self):
        defs = {entry.id: entry for entry in list_preprocessors()}
        canny = defs["canny"]
        self.assertIn("low_threshold", canny.param_schema)
        self.assertEqual(canny.param_schema["low_threshold"].type, "int")

    def test_expanded_preprocessor_catalog_includes_depth_softedge_and_shuffle(self):
        defs = {entry.id: entry for entry in list_preprocessors()}
        for preprocessor_id in (
            "depth-zoe",
            "depth-leres",
            "depth-leres-plus",
            "normal-midas",
            "lineart-standard",
            "shuffle",
            "softedge-hed",
            "softedge-hedsafe",
            "scribble-hed",
            "softedge-pidinet",
            "softedge-pidsafe",
            "scribble-pidinet",
            "mediapipe-face",
            "sam-mobile",
            "sam",
            "teed",
            "anyline",
            "dwpose",
        ):
            self.assertIn(preprocessor_id, defs)

        self.assertEqual(defs["depth-zoe"].param_schema["detect_resolution"].type, "int")
        self.assertEqual(defs["softedge-hedsafe"].defaults["safe"], True)
        self.assertEqual(defs["scribble-pidinet"].defaults["scribble"], True)
        self.assertEqual(defs["mediapipe-face"].param_schema["max_faces"].type, "int")
        self.assertEqual(defs["teed"].param_schema["safe_steps"].type, "int")
        self.assertEqual(defs["anyline"].defaults["detect_resolution"], 1280)

    def test_bool_params_are_coerced_from_form_strings(self):
        preprocessor = get_preprocessor("softedge-hedsafe")
        validated = preprocessor.validate_params(
            {"detect_resolution": "512", "image_resolution": "512", "safe": "false"}
        )
        self.assertEqual(validated["detect_resolution"], 512)
        self.assertFalse(validated["safe"])

    def test_heavy_preprocessors_wire_checkpoint_kwargs(self):
        self.assertEqual(SamMobilePreprocessor.pretrained_model_or_path, "dhkim2810/MobileSAM")
        self.assertEqual(SamMobilePreprocessor.pretrained_kwargs["model_type"], "vit_t")
        self.assertEqual(TEEDPreprocessor.pretrained_kwargs["filename"], "5_model.pth")
        self.assertEqual(AnylinePreprocessor.pretrained_kwargs["subfolder"], "Anyline")

    def test_optional_dependency_availability_is_reported(self):
        preprocessor = get_preprocessor("mediapipe-face")
        with patch(
            "backend.adapters.controlnet_preprocessors.find_spec",
            return_value=None,
        ):
            available, reason, hint = preprocessor.availability()
        self.assertFalse(available)
        self.assertIn("mediapipe", reason)
        self.assertIn("pip install mediapipe", hint)


class ControlNetPreprocessorApiTests(unittest.TestCase):
    def test_list_endpoint_includes_schema_and_compatibility(self):
        response = asyncio.run(list_controlnet_preprocessors())
        canny = next(item for item in response if item.id == "canny")
        self.assertIn("low_threshold", canny.param_schema)
        self.assertGreaterEqual(len(canny.recommended_sd15_control_models), 1)
        depth_zoe = next(item for item in response if item.id == "depth-zoe")
        self.assertIn("detect_resolution", depth_zoe.param_schema)
        self.assertIn(
            "lllyasviel/control_v11f1p_sd15_depth",
            depth_zoe.recommended_sd15_control_models,
        )
        mediapipe_face = next(item for item in response if item.id == "mediapipe-face")
        self.assertIsNotNone(mediapipe_face.available)
        self.assertIn("CrucibleAI/ControlNetMediaPipeFace", mediapipe_face.recommended_sd15_control_models)

    def test_preprocess_endpoint_returns_400_for_invalid_params(self):
        with self.assertRaises(HTTPException) as exc:
            asyncio.run(
                run_controlnet_preprocessor(
                    image=_png_upload(),
                    preprocessor_id="canny",
                    params='{"unexpected": 1}',
                    low_threshold=None,
                    high_threshold=None,
                )
            )
        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("Unsupported params", str(exc.exception.detail))

    def test_preprocess_endpoint_returns_503_for_unavailable_optional_dependency(self):
        with patch(
            "backend.adapters.controlnet_preprocessors.find_spec",
            return_value=None,
        ):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(
                    run_controlnet_preprocessor(
                        image=_png_upload(),
                        preprocessor_id="mediapipe-face",
                        params=None,
                        low_threshold=None,
                        high_threshold=None,
                    )
                )
        self.assertEqual(exc.exception.status_code, 503)
        self.assertIn("mediapipe-face", str(exc.exception.detail))


class ControlNetPreprocessWorkflowTaskTests(unittest.TestCase):
    def test_workflow_requires_object_params(self):
        class _FakePreprocessor:
            def process(self, image, params):
                return image

        with patch("backend.workflow._open_image_ref", return_value=Image.new("RGB", (4, 4))):
            with patch("backend.workflow.get_preprocessor", return_value=_FakePreprocessor()):
                with patch("backend.workflow.save_artifact_png", return_value={"artifact_id": "p0" * 16}):
                    with self.assertRaisesRegex(ValueError, "params must be an object"):
                        _controlnet_preprocess(
                            {
                                "image": {"artifact_id": "a0" * 16},
                                "preprocessor_id": "canny",
                                "params": "{\"low_threshold\": 100}",
                            },
                            _ctx=None,
                        )


if __name__ == "__main__":
    unittest.main()
