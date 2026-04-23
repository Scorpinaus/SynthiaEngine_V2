from dataclasses import dataclass, field
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

    def process(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        merged = {**self.definition.defaults, **params}
        validated = self.validate_params(merged)
        return self.run(image, validated)

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
    detector_instance: Any | None = None

    def _get_detector(self) -> Any:
        if self.detector_instance is None:
            detector_class = _resolve_detector_class(self.detector_names)
            if self.pretrained_kwargs:
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
]


def list_preprocessors() -> list[PreprocessorDefinition]:
    return [preprocessor.definition for preprocessor in _PREPROCESSORS]


def get_preprocessor(preprocessor_id: str) -> BasePreprocessor:
    for preprocessor in _PREPROCESSORS:
        if preprocessor.definition.id == preprocessor_id:
            return preprocessor
    raise KeyError(f"Unknown preprocessor: {preprocessor_id}")
