import torch

from diffusers import (EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler,
    FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler)


class FlowMatchHeunDiscreteSchedulerWithMu(FlowMatchHeunDiscreteScheduler):
    """FlowMatch Heun scheduler variant that supports `mu` via Euler timesteps.

    The upstream Heun scheduler does not expose `mu` in `set_timesteps`. This
    subclass builds an Euler FlowMatch schedule (which supports `mu`) and then
    expands it into a two-stage Heun schedule by repeating intermediate steps.
    """

    def set_timesteps(
        self,
        num_inference_steps: int,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
        **kwargs,
    ):
        """Set inference timesteps and sigmas, including optional FlowMatch `mu`.

        Args:
            num_inference_steps (int): Number of denoising steps.
            device: Target device for stored timestep/sigma tensors.
            sigmas: Optional custom sigma schedule.
            mu: Optional FlowMatch shift parameter passed through Euler setup.
            timesteps: Optional custom timesteps.
            **kwargs: Additional unused keyword arguments for compatibility.
        """
        euler_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.config)
        scheduler_kwargs = {}
        if sigmas is not None:
            scheduler_kwargs["sigmas"] = sigmas
        if timesteps is not None:
            scheduler_kwargs["timesteps"] = timesteps
        if mu is not None:
            scheduler_kwargs["mu"] = mu

        euler_scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)

        self.num_inference_steps = euler_scheduler.num_inference_steps
        base_timesteps = euler_scheduler.timesteps
        base_sigmas = euler_scheduler.sigmas

        timesteps = torch.cat([base_timesteps[:1], base_timesteps[1:].repeat_interleave(2)])
        sigmas = torch.cat([base_sigmas[:1], base_sigmas[1:-1].repeat_interleave(2), base_sigmas[-1:]])

        target_device = device if device is not None else base_sigmas.device
        self.timesteps = timesteps.to(device=target_device)
        self.sigmas = sigmas.to(device=target_device)

        self.prev_derivative = None
        self.dt = None
        self._step_index = None
        self._begin_index = None
        self.sample = None

def create_scheduler(name: str, pipe):
    """Create a scheduler instance from a normalized scheduler name.

    Args:
        name (str): Scheduler identifier (case-insensitive).
        pipe: Diffusers pipeline containing the base scheduler config.

    Returns:
        A configured Diffusers scheduler instance.

    Raises:
        ValueError: If `name` does not match a supported scheduler.
    """
    name = name.lower()
    
    if name == "ddim":
        return DDIMScheduler.from_config(pipe.scheduler.config)
    
    if name == "dpm++2m":
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if name == "dpm++2m_karras":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )    
    if name == "dpm++2m_sde":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
        )
    
    if name == "dpm++2m_sde_karras":
        return DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )        

    if name == "dpm++_sde":
        return DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
        )

    if name == "dpm++_sde_karras":
        return DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True,
        )
        
    if name == "dpm2":
        return KDPM2DiscreteScheduler.from_config(
            pipe.scheduler.config,
        )
    
    if name == "dpm2_karras":
        return KDPM2DiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True,
        )

    if name == "dpm2_a":
        return KDPM2AncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
        )    

    if name == "dpm2_a_karras":
        return KDPM2AncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True,
        ) 
        
    if name == "euler":
        return EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if name == "flowmatch_euler":
        return FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if name == "flowmatch_heun":
        return FlowMatchHeunDiscreteSchedulerWithMu.from_config(pipe.scheduler.config)

    if name == "heun":
        return HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if name == "lms":
        return LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if name == "lms_karras":
        return LMSDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas = True)

    if name == "deis":
        return DEISMultistepScheduler.from_config(pipe.scheduler.config)
    
    if name == "unipc":
        return UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    raise ValueError(f"Unknown scheduler: {name}")
