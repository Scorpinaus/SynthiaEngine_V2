# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from diffusers.loaders import FluxLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import logging
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxModularPipeline(ModularPipeline, FluxLoraLoaderMixin, TextualInversionLoaderMixin):
    """
    A ModularPipeline for Flux.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LowMemoryFluxAutoBlocks"

    def __init__(self, *args, blocks=None, **kwargs):
        if blocks is None:
            from .low_memory import LowMemoryFluxAutoBlocks

            blocks = LowMemoryFluxAutoBlocks()
        super().__init__(*args, blocks=blocks, **kwargs)

    def __call__(self, state=None, output=None, **kwargs):
        sequential_images = kwargs.pop("low_memory_sequential_images", True)
        num_images_per_prompt = self._resolve_num_images_per_prompt(state, kwargs)
        if not self._should_run_sequential_images(
            state=state,
            output=output,
            sequential_images=sequential_images,
            num_images_per_prompt=num_images_per_prompt,
        ):
            return super().__call__(state=state, output=output, **kwargs)

        batch_size = self._infer_sequential_batch_size(kwargs, num_images_per_prompt)
        total_samples = batch_size * num_images_per_prompt
        image_chunks = []
        last_state = None

        for prompt_index in range(batch_size):
            for image_index in range(num_images_per_prompt):
                sample_index = prompt_index * num_images_per_prompt + image_index
                sample_kwargs = self._slice_sequential_sample_kwargs(
                    kwargs,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    batch_size=batch_size,
                    total_samples=total_samples,
                )
                sample_kwargs["num_images_per_prompt"] = 1
                last_state = super().__call__(state=None, output=None, **sample_kwargs)
                self._append_images(image_chunks, last_state.get("images"))

        images = self._combine_images(image_chunks)
        last_state.set("images", images)
        if output is None:
            return last_state
        if isinstance(output, str):
            return last_state.get(output)
        if isinstance(output, (list, tuple)):
            return last_state.get(output)
        raise ValueError(f"Output '{output}' is not a valid output type")

    @staticmethod
    def _resolve_num_images_per_prompt(state, kwargs) -> int:
        if "num_images_per_prompt" in kwargs and kwargs["num_images_per_prompt"] is not None:
            return int(kwargs["num_images_per_prompt"])
        if state is not None and hasattr(state, "get"):
            value = state.get("num_images_per_prompt", 1)
            if value is not None:
                return int(value)
        return 1

    @staticmethod
    def _should_run_sequential_images(*, state, output, sequential_images, num_images_per_prompt: int) -> bool:
        if not sequential_images or state is not None or num_images_per_prompt <= 1:
            return False
        if output is None or output == "images":
            return True
        if isinstance(output, (list, tuple)):
            return set(output) == {"images"}
        return False

    @staticmethod
    def _sequence_len(value) -> int | None:
        if isinstance(value, (str, bytes)):
            return None
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    @classmethod
    def _infer_sequential_batch_size(cls, kwargs, num_images_per_prompt: int) -> int:
        for name in ("prompt", "prompt_2", "image", "resized_image"):
            length = cls._sequence_len(kwargs.get(name))
            if length:
                return int(length)

        for name in ("prompt_embeds", "pooled_prompt_embeds"):
            value = kwargs.get(name)
            if torch.is_tensor(value):
                return int(value.shape[0])

        for name in ("latents", "image_latents"):
            value = kwargs.get(name)
            if torch.is_tensor(value) and value.ndim > 0:
                first_dim = int(value.shape[0])
                if first_dim > 1 and first_dim % num_images_per_prompt == 0:
                    return first_dim // num_images_per_prompt
                return first_dim

        return 1

    @classmethod
    def _slice_sequence_value(cls, value, prompt_index: int, sample_index: int, batch_size: int, total_samples: int):
        length = cls._sequence_len(value)
        if length == total_samples:
            return value[sample_index]
        if length == batch_size:
            return value[prompt_index]
        return value

    @staticmethod
    def _slice_tensor_value(value, prompt_index: int, sample_index: int, batch_size: int, total_samples: int):
        if not torch.is_tensor(value) or value.ndim == 0:
            return value
        if value.shape[0] == total_samples:
            return value[sample_index : sample_index + 1]
        if value.shape[0] == batch_size:
            return value[prompt_index : prompt_index + 1]
        return value

    @classmethod
    def _slice_sequential_sample_kwargs(
        cls,
        kwargs,
        *,
        prompt_index: int,
        sample_index: int,
        batch_size: int,
        total_samples: int,
    ):
        sample_kwargs = dict(kwargs)

        for name in ("prompt", "prompt_2", "image", "resized_image"):
            if name in sample_kwargs:
                sample_kwargs[name] = cls._slice_sequence_value(
                    sample_kwargs[name], prompt_index, sample_index, batch_size, total_samples
                )

        for name in ("prompt_embeds", "pooled_prompt_embeds", "latents", "image_latents"):
            if name in sample_kwargs:
                sample_kwargs[name] = cls._slice_tensor_value(
                    sample_kwargs[name], prompt_index, sample_index, batch_size, total_samples
                )

        if "generator" in sample_kwargs:
            sample_kwargs["generator"] = cls._slice_sequence_value(
                sample_kwargs["generator"], prompt_index, sample_index, batch_size, total_samples
            )

        return sample_kwargs

    @staticmethod
    def _append_images(image_chunks, images) -> None:
        if isinstance(images, list):
            image_chunks.extend(images)
        else:
            image_chunks.append(images)

    @staticmethod
    def _combine_images(image_chunks):
        if not image_chunks:
            return []
        if all(torch.is_tensor(chunk) for chunk in image_chunks):
            return torch.cat(image_chunks, dim=0)
        if all(isinstance(chunk, np.ndarray) for chunk in image_chunks):
            return np.concatenate(image_chunks, axis=0)
        return image_chunks

    @property
    def default_height(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_width(self):
        return self.default_sample_size * self.vae_scale_factor

    @property
    def default_sample_size(self):
        return 128

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if getattr(self, "vae", None) is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 16
        if getattr(self, "transformer", None):
            num_channels_latents = self.transformer.config.in_channels // 4
        return num_channels_latents


class FluxKontextModularPipeline(FluxModularPipeline):
    """
    A ModularPipeline for Flux Kontext.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LowMemoryFluxKontextAutoBlocks"

    def __init__(self, *args, blocks=None, **kwargs):
        if blocks is None:
            from .low_memory import LowMemoryFluxKontextAutoBlocks

            blocks = LowMemoryFluxKontextAutoBlocks()
        super().__init__(*args, blocks=blocks, **kwargs)
