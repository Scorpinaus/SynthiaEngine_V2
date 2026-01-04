from diffusers import (EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler)

def create_scheduler(name: str, pipe):
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
