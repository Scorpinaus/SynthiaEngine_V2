"""Sampling helpers for the PixelDiT Modular Diffusers prototype."""

from __future__ import annotations

import torch


class NoiseScheduleFlow:
    """PixelDiT's flow-matching schedule used by the official FlowDPM sampler."""

    T = 1.0
    total_N = 1000

    @staticmethod
    def marginal_alpha(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    @staticmethod
    def marginal_std(t: torch.Tensor) -> torch.Tensor:
        return t

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(self.marginal_alpha(t))

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        return self.marginal_log_mean_coeff(t) - torch.log(self.marginal_std(t))


def _expand_dims(value: torch.Tensor, dims: int) -> torch.Tensor:
    if value.dim() == 0:
        value = value[None]
    return value.reshape(value.shape[0], *([1] * (dims - 1)))


def _inverse_flow_lambda(value: torch.Tensor) -> torch.Tensor:
    # Matches NVIDIA PixelDiT's FlowDPM inverse_lambda implementation.
    return torch.exp(-value)


def _time_uniform_flow_steps(
    *,
    steps: int,
    device: torch.device,
    flow_shift: float,
    t_start: float = 1.0,
    t_end: float = 1.0 / 1000.0,
) -> torch.Tensor:
    betas = torch.linspace(t_start, t_end, steps + 1, device=device)
    sigmas = 1.0 - betas
    shifted = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)
    return shifted.flip(dims=[0])


def _first_order_update(
    *,
    x: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    model_s: torch.Tensor,
    schedule: NoiseScheduleFlow,
) -> torch.Tensor:
    lambda_s = schedule.marginal_lambda(s)
    lambda_t = schedule.marginal_lambda(t)
    h = lambda_t - lambda_s
    sigma_s = schedule.marginal_std(s)
    sigma_t = schedule.marginal_std(t)
    alpha_t = schedule.marginal_alpha(t)
    phi_1 = torch.expm1(-h)
    return (_expand_dims(sigma_t / sigma_s, x.dim()) * x) - (
        _expand_dims(alpha_t * phi_1, x.dim()) * model_s
    )


def _second_order_singlestep_update(
    *,
    x: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    model_s: torch.Tensor,
    model_fn,
    schedule: NoiseScheduleFlow,
    r1: float = 0.5,
) -> torch.Tensor:
    lambda_s = schedule.marginal_lambda(s)
    lambda_t = schedule.marginal_lambda(t)
    h = lambda_t - lambda_s
    s1 = _inverse_flow_lambda(lambda_s + r1 * h)
    sigma_s = schedule.marginal_std(s)
    sigma_s1 = schedule.marginal_std(s1)
    sigma_t = schedule.marginal_std(t)
    alpha_s1 = schedule.marginal_alpha(s1)
    alpha_t = schedule.marginal_alpha(t)
    phi_11 = torch.expm1(-r1 * h)
    phi_1 = torch.expm1(-h)
    x_s1 = (_expand_dims(sigma_s1 / sigma_s, x.dim()) * x) - (
        _expand_dims(alpha_s1 * phi_11, x.dim()) * model_s
    )
    model_s1 = model_fn(x_s1, s1)
    return (
        (_expand_dims(sigma_t / sigma_s, x.dim()) * x)
        - (_expand_dims(alpha_t * phi_1, x.dim()) * model_s)
        - (_expand_dims((0.5 / r1) * alpha_t * phi_1, x.dim()) * (model_s1 - model_s))
    )


def _second_order_multistep_update(
    *,
    x: torch.Tensor,
    model_prev_1: torch.Tensor,
    model_prev_0: torch.Tensor,
    t_prev_1: torch.Tensor,
    t_prev_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseScheduleFlow,
) -> torch.Tensor:
    lambda_prev_1 = schedule.marginal_lambda(t_prev_1)
    lambda_prev_0 = schedule.marginal_lambda(t_prev_0)
    lambda_t = schedule.marginal_lambda(t)
    sigma_prev_0 = schedule.marginal_std(t_prev_0)
    sigma_t = schedule.marginal_std(t)
    alpha_t = schedule.marginal_alpha(t)
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    d1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h)
    return (
        (_expand_dims(sigma_t / sigma_prev_0, x.dim()) * x)
        - (_expand_dims(alpha_t * phi_1, x.dim()) * model_prev_0)
        - (_expand_dims(0.5 * alpha_t * phi_1, x.dim()) * d1_0)
    )


def flow_dpm_sample(
    transformer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    negative_attention_mask: torch.Tensor | None,
    *,
    num_inference_steps: int,
    guidance_scale: float,
    flow_shift: float,
    interval_guidance: tuple[float, float],
) -> torch.Tensor:
    """Run PixelDiT's FlowDPM sampling path.

    This ports the PixelDiT flow-specific DPM-Solver++ path used by NVIDIA's
    public inference code: discrete flow schedule, ``time_uniform_flow`` step
    spacing, order-2 multistep updates, and classifier-free guidance.
    """

    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    schedule = NoiseScheduleFlow()
    device = latents.device
    prompt_embeds = prompt_embeds.to(device=device, dtype=latents.dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=latents.dtype)
    # Official PixelDiT inference calls forward_with_dpmsolver with mask=None.
    # The architecture has partial mask plumbing, but the same mask is also
    # forwarded into the pixel attention path, where tokenizer masks are invalid.
    attention_mask = None
    negative_attention_mask = None

    def noise_pred_fn(x: torch.Tensor, t_continuous: torch.Tensor, cond: torch.Tensor, mask) -> torch.Tensor:
        if t_continuous.dim() == 0:
            t_continuous = t_continuous.expand(x.shape[0])
        t_input = t_continuous * schedule.total_N
        velocity = transformer.forward_with_dpmsolver(x, t_input, cond, mask=mask)
        sigma_t = schedule.marginal_std(t_continuous)
        return (_expand_dims(1.0 - sigma_t, x.dim()).to(x) * velocity) + x

    def guided_noise_fn(x: torch.Tensor, t_continuous: torch.Tensor) -> torch.Tensor:
        start_guidance, end_guidance = interval_guidance
        use_cfg = (
            negative_prompt_embeds is not None
            and guidance_scale != 1.0
            and start_guidance < float(t_continuous.reshape(-1)[0]) < end_guidance
        )
        if not use_cfg:
            return noise_pred_fn(x, t_continuous, prompt_embeds, attention_mask)
        x_in = torch.cat([x, x])
        t_in = t_continuous.expand(x_in.shape[0]) if t_continuous.dim() == 0 else torch.cat([t_continuous, t_continuous])
        cond_in = torch.cat([negative_prompt_embeds, prompt_embeds])
        mask_in = None
        if negative_attention_mask is not None and attention_mask is not None:
            mask_in = torch.cat([negative_attention_mask, attention_mask])
        noise_uncond, noise = noise_pred_fn(x_in, t_in, cond_in, mask_in).chunk(2)
        return noise_uncond + guidance_scale * (noise - noise_uncond)

    def data_prediction_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = guided_noise_fn(x, t)
        alpha_t = schedule.marginal_alpha(t)
        sigma_t = schedule.marginal_std(t)
        return (x - _expand_dims(sigma_t, x.dim()) * noise) / _expand_dims(alpha_t, x.dim())

    timesteps = _time_uniform_flow_steps(steps=num_inference_steps, device=device, flow_shift=flow_shift)
    step_order = 2 if num_inference_steps >= 2 else 1
    t_prev_list = [timesteps[0]]
    model_prev_list = [data_prediction_fn(latents, timesteps[0])]
    latents = _first_order_update(
        x=latents,
        s=timesteps[0],
        t=timesteps[1],
        model_s=model_prev_list[0],
        schedule=schedule,
    )
    if num_inference_steps == 1:
        return latents

    t_prev_list.append(timesteps[1])
    model_prev_list.append(data_prediction_fn(latents, timesteps[1]))
    for step_index in range(2, num_inference_steps + 1):
        t = timesteps[step_index]
        order = min(step_order, num_inference_steps + 1 - step_index)
        if order == 1:
            latents = _first_order_update(
                x=latents,
                s=t_prev_list[-1],
                t=t,
                model_s=model_prev_list[-1],
                schedule=schedule,
            )
        else:
            latents = _second_order_multistep_update(
                x=latents,
                model_prev_1=model_prev_list[-2],
                model_prev_0=model_prev_list[-1],
                t_prev_1=t_prev_list[-2],
                t_prev_0=t_prev_list[-1],
                t=t,
                schedule=schedule,
            )
        t_prev_list = [t_prev_list[-1], t]
        if step_index < num_inference_steps:
            model_prev_list = [model_prev_list[-1], data_prediction_fn(latents, t)]
    return latents


flow_euler_sample = flow_dpm_sample
