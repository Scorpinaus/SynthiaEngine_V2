from dataclasses import dataclass
from typing import Any

from typing import Callable

from PIL import Image


@dataclass(frozen=True)
class PreprocessorDefinition:
    id: str
    name: str
    description: str
    defaults: dict[str, Any]


class BasePreprocessor:
    definition: PreprocessorDefinition

    def process(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        merged = {**self.definition.defaults, **params}
        return self.run(image, merged)

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


class ControlNetAuxPreprocessor(BasePreprocessor):
    detector_names: list[str]
    detector_instance: Any | None = None

    def _get_detector(self) -> Any:
        if self.detector_instance is None:
            detector_class = _resolve_detector_class(self.detector_names)
            self.detector_instance = detector_class()
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
    )


class HEDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["HEDdetector", "HEDDetector"]
    definition = PreprocessorDefinition(
        id="hed",
        name="HED",
        description="Holistically-nested edge detection preprocessor.",
        defaults={},
    )


class MidasDepthPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["MidasDetector"]
    definition = PreprocessorDefinition(
        id="midas-depth",
        name="Midas Depth",
        description="Predicts depth maps using MiDaS.",
        defaults={},
    )


class OpenPosePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["OpenposeDetector", "OpenPoseDetector"]
    definition = PreprocessorDefinition(
        id="openpose",
        name="OpenPose",
        description="Detects human pose keypoints.",
        defaults={},
    )


class MLSDPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["MLSDdetector", "MLSDDetector"]
    definition = PreprocessorDefinition(
        id="mlsd",
        name="MLSD",
        description="Detects straight lines using MLSD.",
        defaults={},
    )


class LineartPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LineartDetector"]
    definition = PreprocessorDefinition(
        id="lineart",
        name="Lineart",
        description="Extracts lineart from the input image.",
        defaults={},
    )


class LineartAnimePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["LineartAnimeDetector"]
    definition = PreprocessorDefinition(
        id="lineart-anime",
        name="Lineart Anime",
        description="Extracts anime-style lineart from the input image.",
        defaults={},
    )


class PidiNetPreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["PidiNetDetector"]
    definition = PreprocessorDefinition(
        id="pidinet",
        name="PidiNet",
        description="Edge detection using PiDiNet.",
        defaults={},
    )


class NormalBaePreprocessor(ControlNetAuxPreprocessor):
    detector_names = ["NormalBaeDetector"]
    definition = PreprocessorDefinition(
        id="normal-bae",
        name="Normal BAE",
        description="Predicts surface normals (NormalBae).",
        defaults={},
    )

    def run(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        low_threshold = int(params.get("low_threshold", 100) or 100)
        high_threshold = int(params.get("high_threshold", 200) or 200)
        detector = CannyDetector()
        return detector(image, low_threshold=low_threshold, high_threshold=high_threshold)


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
