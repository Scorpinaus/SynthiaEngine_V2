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
from typing import Any, Dict, List, Optional, Union

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AsymmetricAutoencoderKL, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging

from pipeline_stable_diffusion import StableDiffusionPipeline as _Text2ImgPipeline
from pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline as _Img2ImgPipeline
from pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline as _InpaintPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionUnifiedPipeline(_Text2ImgPipeline):
    """
    Unified Stable Diffusion pipeline supporting text2img, img2img, and inpaint modes.

    Use `mode="text2img"`, `mode="img2img"`, or `mode="inpaint"` to select the behavior.
    """

    def __init__(
        self,
        vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        mode: str = "text2img",
        strength: float = 0.8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        padding_mask_crop: Optional[int] = None,
        masked_image_latents=None,
        **kwargs,
    ):
        mode = (mode or "text2img").lower()
        if mode not in ("text2img", "img2img", "inpaint"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'text2img', 'img2img', or 'inpaint'.")

        if mode != "text2img" and guidance_rescale not in (0.0, None):
            logger.warning("guidance_rescale is currently only applied for text2img.")

        if mode == "text2img":
            return _Text2ImgPipeline.__call__(
                self,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )

        if mode == "img2img":
            return _Img2ImgPipeline.__call__(
                self,
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )

        return _InpaintPipeline.__call__(
            self,
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            padding_mask_crop=padding_mask_crop,
            masked_image_latents=masked_image_latents,
            **kwargs,
        )

    def text2img(self, *args, **kwargs):
        return self.__call__(*args, mode="text2img", **kwargs)

    def img2img(self, *args, **kwargs):
        return self.__call__(*args, mode="img2img", **kwargs)

    def inpaint(self, *args, **kwargs):
        return self.__call__(*args, mode="inpaint", **kwargs)
