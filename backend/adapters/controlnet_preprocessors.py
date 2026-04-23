from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any
from typing import Literal

from typing import Callable

from PIL import Image


@dataclass(frozen=True)
class PreprocessorParamSpec:
    type: Literal["int", "float", "bool", "str"]
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class PreprocessorDefinition:
    id: str
    name: str
    description: str
    defaults: dict[str, Any]
    param_schema: dict[str, PreprocessorParamSpec] = field(default_factory=dict)


class BasePreprocessor:
    definition: PreprocessorDefinition
    required_modules: dict[str, str] = {}

    def process(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        self.require_available()
        merged = {**self.definition.defaults, **params}
        validated = self.validate_params(merged)
        return self.run(image, validated)

    def availability(self) -> tuple[bool, str | None, str | None]:
        for module_name, install_hint in self.required_modules.items():
            if find_spec(module_name) is None:
                return (
                    False,
                    f"Optional dependency '{module_name}' is not installed.",
                    install_hint,
                )
        return True, None, None

    def require_available(self) -> None:
        available, reason, install_hint = self.availability()
        if not available:
            detail = f"Preprocessor '{self.definition.id}' is unavailable: {reason}"
            if install_hint:
                detail = f"{detail} {install_hint}"
            raise RuntimeError(detail)

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ValueError("params must be an object")

        schema = self.definition.param_schema or {}
        allowed_keys = set(schema.keys())
        unknown = sorted(set(params.keys()) - allowed_keys)
        if unknown:
            allowed = ", ".join(sorted(allowed_keys)) if allowed_keys else "<none>"
            raise ValueError(
                f"Unsupported params for preprocessor '{self.definition.id}': "
                f"{', '.join(unknown)}. Allowed params: {allowed}."
            )

        validated: dict[str, Any] = {}
        for key, value in params.items():
            spec = schema[key]
            coerced = _coerce_param_value(key, value, spec)
            if spec.type in {"int", "float"}:
                numeric = float(coerced)
                if spec.minimum is not None and numeric < spec.minimum:
                    raise ValueError(
                        f"Param '{key}' must be >= {spec.minimum} for '{self.definition.id}'."
                    )
                if spec.maximum is not None and numeric > spec.maximum:
                    raise ValueError(
                        f"Param '{key}' must be <= {spec.maximum} for '{self.definition.id}'."
                    )
            validated[key] = coerced
        return validated

    def run(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        raise NotImplementedError


def _resolve_detector_class(names: list[str]) -> Callable[..., Any]:
    try:
        import controlnet_aux  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "controlnet-aux is required for ControlNet preprocessors."
        ) from exc

    for name in names:
        detector_class = getattr(controlnet_aux, name, None)
        if detector_class is not None:
            return detector_class

    raise RuntimeError(f"Could not resolve controlnet-aux detector: {names}")


def _build_detector(detector_class: type[Any], pretrained_model_or_path: str | None) -> Any:
    if pretrained_model_or_path and hasattr(detector_class, "from_pretrained"):
        try:
            return detector_class.from_pretrained(pretrained_model_or_path)
        except Exception as exc:  # pragma: no cover - external model loading failures
            raise RuntimeError(
                f"Failed to load detector from '{pretrained_model_or_path}'."
            ) from exc

    try:
        return detector_class()
    except TypeError:
        if hasattr(detector_class, "from_pretrained"):
            try:
                return detector_class.from_pretrained("lllyasviel/Annotators")
            except Exception as exc:  # pragma: no cover - external model loading failures
                raise RuntimeError(
                    "Detector requires pretrained weights but failed to load default "
                    "'lllyasviel/Annotators'."
                ) from exc
        raise


def _coerce_param_value(key: str, value: Any, spec: PreprocessorParamSpec) -> Any:
    expected = spec.type
    if expected == "int":
        if isinstance(value, bool):
            raise ValueError(f"Param '{key}' must be an int.")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"Param '{key}' must be an int.")
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ValueError(f"Param '{key}' must be an int.") from exc
        raise ValueError(f"Param '{key}' must be an int.")

    if expected == "float":
        if isinstance(value, bool):
            raise ValueError(f"Param '{key}' must be a float.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise ValueError(f"Param '{key}' must be a float.") from exc
        raise ValueError(f"Param '{key}' must be a float.")

    if expected == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"Param '{key}' must be a bool.")

    if expected == "str":
        if isinstance(value, str):
            return value
        raise ValueError(f"Param '{key}' must be a string.")

    raise ValueError(f"Unsupported param type '{expected}' for '{key}'.")


class ControlNetAuxPreprocessor(BasePreprocessor):
    detector_names: list[str]
    pretrained_model_or_path: str | None = None
    pretrained_kwargs: dict[str, Any] = {}
    detector_init_kwargs: dict[str, Any] = {}
    detector_instance: Any | None = None

    def _get_detector(self) -> Any:
        if self.detector_instance is None:
            detector_class = _resolve_detector_class(self.detector_names)
            if self.pretrained_kwargs:
                if not hasattr(detector_class, "from_pretrained"):
                    raise RuntimeError(
                        f"Detector for '{self.definition.id}' does not support "
                        "from_pretrained loading."
                    )
                try:
                    self.detector_instance = detector_class.from_pretrained(
                        self.pretrained_model_or_path,
                        **self.pretrained_kwargs,
                    )
                except Exception as exc:  # pragma: no cover - external model loading failures
                    raise RuntimeError(
                        f"Failed to load detector from '{self.pretrained_model_or_path}'."
                    ) from exc
            else:
                if self.detector_init_kwargs:
                    try:
                        self.detector_instance = detector_class(
                            **self.detector_init_kwargs
                        )
                    except Exception as exc:  # pragma: no cover - external init failures
                        raise RuntimeError(
                            f"Failed to initialize detector for '{self.definition.id}'."
                        ) from exc
                else:
                    self.detector_instance = _build_detector(
                        detector_class, self.pretrained_model_or_path
                    )
        return self.detector_instance

    def run(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        detector = self._get_detector()
        return detector(image, **params)


def _resolution_params() -> dict[str, PreprocessorParamSpec]:
    return {
        "detect_resolution": PreprocessorParamSpec(
            type="int",
            description="Resolution used by the detector before resizing.",
            minimum=64,
            maximum=4096,
        ),
        "image_resolution": PreprocessorParamSpec(
            type="int",
            description="Output resolution used by the detector.",
            minimum=64,
            maximum=4096,
        ),
    }


def _safe_param() -> dict[str, PreprocessorParamSpec]:
    return {
        "safe": PreprocessorParamSpec(
            type="bool",
            description="Use safer edge post-processing.",
        )
    }


def _scribble_param() -> dict[str, PreprocessorParamSpec]:
    return {
        "scribble": PreprocessorParamSpec(
            type="bool",
            description="Convert detected edges into a scribble-style control image.",
        )
    }


class CannyPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["CannyDetector"]
    definition = PreprocessorDefinition(
        id="canny",
        name="Canny",
        description="Detects edges with the Canny algorithm using controlnet-aux.",
        defaults={"low_threshold": 100, "high_threshold": 200},
        param_schema={
            "low_threshold": PreprocessorParamSpec(
                type="int",
                description="Lower Canny threshold.",
                minimum=0,
                maximum=255,
            ),
            "high_threshold": PreprocessorParamSpec(
                type="int",
                description="Upper Canny threshold.",
                minimum=0,
                maximum=255,
            ),
        },
    )


class HEDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["HEDdetector", "HEDDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="hed",
        name="HED",
        description="Holistically-nested edge detection preprocessor.",
        defaults={},
        param_schema={},
    )


class SoftedgeHEDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["HEDdetector", "HEDDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="softedge-hed",
        name="SoftEdge HED",
        description="Soft edge detection using HED.",
        defaults={"detect_resolution": 512, "image_resolution": 512, "safe": False},
        param_schema={**_resolution_params(), **_safe_param()},
    )


class SoftedgeHEDSafePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["HEDdetector", "HEDDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="softedge-hedsafe",
        name="SoftEdge HED Safe",
        description="Soft edge detection using HED safe mode.",
        defaults={"detect_resolution": 512, "image_resolution": 512, "safe": True},
        param_schema={**_resolution_params(), **_safe_param()},
    )


class ScribbleHEDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["HEDdetector", "HEDDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="scribble-hed",
        name="Scribble HED",
        description="Scribble-style edge extraction using HED.",
        defaults={
            "detect_resolution": 512,
            "image_resolution": 512,
            "safe": False,
            "scribble": True,
        },
        param_schema={**_resolution_params(), **_safe_param(), **_scribble_param()},
    )


class MidasDepthPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["MidasDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="midas-depth",
        name="Midas Depth",
        description="Predicts depth maps using MiDaS.",
        defaults={},
        param_schema={},
    )


class MidasNormalPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["MidasDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="normal-midas",
        name="Midas Normal",
        description="Predicts surface normals from MiDaS depth.",
        defaults={
            "detect_resolution": 512,
            "image_resolution": 512,
            "depth_and_normal": True,
            "bg_th": 0.1,
        },
        param_schema={
            **_resolution_params(),
            "depth_and_normal": PreprocessorParamSpec(
                type="bool",
                description="Return the MiDaS normal-map output.",
            ),
            "bg_th": PreprocessorParamSpec(
                type="float",
                description="Background threshold for normal-map generation.",
                minimum=0.0,
                maximum=1.0,
            ),
        },
    )

    def run(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        detector = self._get_detector()
        _depth_image, normal_image = detector(image, **params)
        return normal_image


class ZoeDepthPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["ZoeDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="depth-zoe",
        name="Zoe Depth",
        description="Predicts depth maps using ZoeDepth.",
        defaults={"detect_resolution": 512, "image_resolution": 512},
        param_schema=_resolution_params(),
    )


class LeresDepthPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LeresDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="depth-leres",
        name="LeReS Depth",
        description="Predicts depth maps using LeReS.",
        defaults={
            "detect_resolution": 512,
            "image_resolution": 512,
            "thr_a": 0,
            "thr_b": 0,
            "boost": False,
        },
        param_schema={
            **_resolution_params(),
            "thr_a": PreprocessorParamSpec(
                type="int",
                description="Near depth threshold.",
                minimum=0,
                maximum=255,
            ),
            "thr_b": PreprocessorParamSpec(
                type="int",
                description="Far depth threshold.",
                minimum=0,
                maximum=255,
            ),
            "boost": PreprocessorParamSpec(
                type="bool",
                description="Use boosted LeReS depth estimation.",
            ),
        },
    )


class LeresDepthBoostPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LeresDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="depth-leres-plus",
        name="LeReS Depth Plus",
        description="Predicts boosted depth maps using LeReS.",
        defaults={
            "detect_resolution": 512,
            "image_resolution": 512,
            "thr_a": 0,
            "thr_b": 0,
            "boost": True,
        },
        param_schema=LeresDepthPreprocessor.definition.param_schema,
    )


class OpenPosePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["OpenposeDetector", "OpenPoseDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="openpose",
        name="OpenPose",
        description="Detects human pose keypoints.",
        defaults={},
        param_schema={},
    )


class MLSDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["MLSDdetector", "MLSDDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="mlsd",
        name="MLSD",
        description="Detects straight lines using MLSD.",
        defaults={},
        param_schema={},
    )


class LineartPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LineartDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="lineart",
        name="Lineart",
        description="Extracts lineart from the input image.",
        defaults={},
        param_schema={},
    )


class LineartAnimePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LineartAnimeDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="lineart-anime",
        name="Lineart Anime",
        description="Extracts anime-style lineart from the input image.",
        defaults={},
        param_schema={},
    )


class LineartStandardPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LineartStandardDetector"]
    definition = PreprocessorDefinition(
        id="lineart-standard",
        name="Lineart Standard",
        description="Extracts standard lineart without pretrained weights.",
        defaults={
            "detect_resolution": 512,
            "guassian_sigma": 6.0,
            "intensity_threshold": 8,
        },
        param_schema={
            "detect_resolution": PreprocessorParamSpec(
                type="int",
                description="Resolution used by the detector.",
                minimum=64,
                maximum=4096,
            ),
            "guassian_sigma": PreprocessorParamSpec(
                type="float",
                description="Gaussian blur sigma used by the detector.",
                minimum=0.0,
                maximum=32.0,
            ),
            "intensity_threshold": PreprocessorParamSpec(
                type="int",
                description="Line intensity threshold.",
                minimum=0,
                maximum=255,
            ),
        },
    )


class ContentShufflePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["ContentShuffleDetector"]
    definition = PreprocessorDefinition(
        id="shuffle",
        name="Content Shuffle",
        description="Shuffles image content while preserving color and texture cues.",
        defaults={"detect_resolution": 512, "image_resolution": 512},
        param_schema=_resolution_params(),
    )


class PidiNetPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["PidiNetDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="pidinet",
        name="PidiNet",
        description="Edge detection using PiDiNet.",
        defaults={},
        param_schema={},
    )


class SoftedgePidiNetPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["PidiNetDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="softedge-pidinet",
        name="SoftEdge PidiNet",
        description="Soft edge detection using PiDiNet.",
        defaults={"detect_resolution": 512, "image_resolution": 512, "safe": False},
        param_schema={**_resolution_params(), **_safe_param()},
    )


class SoftedgePidiNetSafePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["PidiNetDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="softedge-pidsafe",
        name="SoftEdge PidiNet Safe",
        description="Soft edge detection using PiDiNet safe mode.",
        defaults={"detect_resolution": 512, "image_resolution": 512, "safe": True},
        param_schema={**_resolution_params(), **_safe_param()},
    )


class ScribblePidiNetPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["PidiNetDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="scribble-pidinet",
        name="Scribble PidiNet",
        description="Scribble-style edge extraction using PiDiNet.",
        defaults={
            "detect_resolution": 512,
            "image_resolution": 512,
            "safe": False,
            "scribble": True,
        },
        param_schema={**_resolution_params(), **_safe_param(), **_scribble_param()},
    )


class NormalBaePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["NormalBaeDetector"]
    pretrained_model_or_path = "lllyasviel/Annotators"
    definition = PreprocessorDefinition(
        id="normal-bae",
        name="Normal BAE",
        description="Predicts surface normals (NormalBae).",
        defaults={},
        param_schema={},
    )


class MediaPipeFacePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["MediapipeFaceDetector"]
    required_modules = {
        "mediapipe": "Install optional MediaPipe support with `pip install mediapipe`."
    }
    definition = PreprocessorDefinition(
        id="mediapipe-face",
        name="MediaPipe Face",
        description="Detects face mesh landmarks using MediaPipe.",
        defaults={
            "max_faces": 1,
            "min_confidence": 0.5,
            "detect_resolution": 512,
            "image_resolution": 512,
        },
        param_schema={
            **_resolution_params(),
            "max_faces": PreprocessorParamSpec(
                type="int",
                description="Maximum number of faces to annotate.",
                minimum=1,
                maximum=16,
            ),
            "min_confidence": PreprocessorParamSpec(
                type="float",
                description="Minimum face detection confidence.",
                minimum=0.0,
                maximum=1.0,
            ),
        },
    )


class SamMobilePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["SamDetector"]
    pretrained_model_or_path = "dhkim2810/MobileSAM"
    pretrained_kwargs = {"model_type": "vit_t", "filename": "mobile_sam.pt"}
    definition = PreprocessorDefinition(
        id="sam-mobile",
        name="SAM Mobile",
        description="Segments image regions using the smaller MobileSAM checkpoint.",
        defaults={"detect_resolution": 512, "image_resolution": 512},
        param_schema=_resolution_params(),
    )


class SamPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["SamDetector"]
    pretrained_model_or_path = "ybelkada/segment-anything"
    pretrained_kwargs = {"subfolder": "checkpoints"}
    definition = PreprocessorDefinition(
        id="sam",
        name="SAM",
        description="Segments image regions using Segment Anything.",
        defaults={"detect_resolution": 512, "image_resolution": 512},
        param_schema=_resolution_params(),
    )


class TEEDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["TEEDdetector"]
    pretrained_model_or_path = "fal-ai/teed"
    pretrained_kwargs = {"filename": "5_model.pth"}
    definition = PreprocessorDefinition(
        id="teed",
        name="TEED",
        description="Thin edge extraction using the TEED checkpoint.",
        defaults={"detect_resolution": 512, "safe_steps": 2},
        param_schema={
            "detect_resolution": PreprocessorParamSpec(
                type="int",
                description="Resolution used by the detector.",
                minimum=64,
                maximum=4096,
            ),
            "safe_steps": PreprocessorParamSpec(
                type="int",
                description="Edge refinement safety steps.",
                minimum=0,
                maximum=10,
            ),
        },
    )


class AnylinePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["AnylineDetector"]
    pretrained_model_or_path = "TheMistoAI/MistoLine"
    pretrained_kwargs = {"filename": "MTEED.pth", "subfolder": "Anyline"}
    definition = PreprocessorDefinition(
        id="anyline",
        name="Anyline",
        description="Hybrid line extraction using the MistoLine Anyline checkpoint.",
        defaults={
            "detect_resolution": 1280,
            "guassian_sigma": 2.0,
            "intensity_threshold": 3,
        },
        param_schema={
            "detect_resolution": PreprocessorParamSpec(
                type="int",
                description="Resolution used by the detector.",
                minimum=64,
                maximum=4096,
            ),
            "guassian_sigma": PreprocessorParamSpec(
                type="float",
                description="Gaussian blur sigma used by the detector.",
                minimum=0.0,
                maximum=32.0,
            ),
            "intensity_threshold": PreprocessorParamSpec(
                type="int",
                description="Line intensity threshold.",
                minimum=0,
                maximum=255,
            ),
        },
    )


class DWPosePreprocessor(BasePreprocessor):
    required_modules = {
        "easy_dwpose": "Install optional DWPose support with `pip install easy-dwpose`."
    }
    detector_instance: Any | None = None
    definition = PreprocessorDefinition(
        id="dwpose",
        name="DWPose",
        description="Whole-body pose detection with body, hand, and face keypoints.",
        defaults={"include_hands": True, "include_face": True},
        param_schema={
            "include_hands": PreprocessorParamSpec(
                type="bool",
                description="Include hand keypoints in the pose map.",
            ),
            "include_face": PreprocessorParamSpec(
                type="bool",
                description="Include face keypoints in the pose map.",
            ),
        },
    )

    def _get_detector(self) -> Any:
        if self.detector_instance is None:
            try:
                import torch
                from easy_dwpose import DWposeDetector
            except ImportError as exc:  # pragma: no cover - guarded by availability
                raise RuntimeError(
                    "DWPose requires `easy-dwpose`. Install it and restart the backend."
                ) from exc

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.detector_instance = DWposeDetector(device=device)
        return self.detector_instance

    def run(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        detector = self._get_detector()
        return detector(image, output_type="pil", **params)


_PREPROCESSORS: list[BasePreprocessor] = [
    CannyPreprocessor(),
    HEDPreprocessor(),
    SoftedgeHEDPreprocessor(),
    SoftedgeHEDSafePreprocessor(),
    ScribbleHEDPreprocessor(),
    MidasDepthPreprocessor(),
    MidasNormalPreprocessor(),
    ZoeDepthPreprocessor(),
    LeresDepthPreprocessor(),
    LeresDepthBoostPreprocessor(),
    OpenPosePreprocessor(),
    MLSDPreprocessor(),
    LineartPreprocessor(),
    LineartAnimePreprocessor(),
    LineartStandardPreprocessor(),
    ContentShufflePreprocessor(),
    PidiNetPreprocessor(),
    SoftedgePidiNetPreprocessor(),
    SoftedgePidiNetSafePreprocessor(),
    ScribblePidiNetPreprocessor(),
    NormalBaePreprocessor(),
    MediaPipeFacePreprocessor(),
    SamMobilePreprocessor(),
    SamPreprocessor(),
    TEEDPreprocessor(),
    AnylinePreprocessor(),
    DWPosePreprocessor(),
]


def list_preprocessors() -> list[PreprocessorDefinition]:
    return [preprocessor.definition for preprocessor in _PREPROCESSORS]


def get_preprocessor(preprocessor_id: str) -> BasePreprocessor:
    for preprocessor in _PREPROCESSORS:
        if preprocessor.definition.id == preprocessor_id:
            return preprocessor
    raise KeyError(f"Unknown preprocessor: {preprocessor_id}")
