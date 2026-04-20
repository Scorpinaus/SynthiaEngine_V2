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
    detector_instance: Any | None = None

    def _get_detector(self) -> Any:
        if self.detector_instance is None:
            detector_class = _resolve_detector_class(self.detector_names)
            self.detector_instance = _build_detector(
                detector_class, self.pretrained_model_or_path
            )
        return self.detector_instance

    def run(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        detector = self._get_detector()
        return detector(image, **params)


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
    MidasDepthPreprocessor(),
    OpenPosePreprocessor(),
    MLSDPreprocessor(),
    LineartPreprocessor(),
    LineartAnimePreprocessor(),
    PidiNetPreprocessor(),
    NormalBaePreprocessor(),
]


def list_preprocessors() -> list[PreprocessorDefinition]:
    return [preprocessor.definition for preprocessor in _PREPROCESSORS]


def get_preprocessor(preprocessor_id: str) -> BasePreprocessor:
    for preprocessor in _PREPROCESSORS:
        if preprocessor.definition.id == preprocessor_id:
            return preprocessor
    raise KeyError(f"Unknown preprocessor: {preprocessor_id}")
