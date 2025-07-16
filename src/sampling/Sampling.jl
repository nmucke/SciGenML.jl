"""
    Sampling

Module for sampling from models.

This module contains functions for sampling using generative models.
"""

module Sampling

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.ZERO_TOL as ZERO_TOL
import SciGenML.DEFAULT_DEVICE as DEFAULT_DEVICE
import SciGenML.Models as Models
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators

import Random
import Distributions
import ProgressBars
import Lux

"""
    sample(
        model,
        args...; 
        kwargs...
    )

    Generic training function for all models.
"""
function sample(model, args...; kwargs...)
    return sample(model.trait, model, args...; kwargs...)
end

##### Stochastic Interpolant Sampling #####
include("sample_stochastic_interpolant.jl")
include("posterior_sample_stochastic_interpolant.jl")

##### Flow Matching Sampling #####
include("sample_flow_matching.jl")

##### Denoising Diffusion Model Sampling #####
include("sample_diffusion_model.jl")

export sample
export posterior_sample

end
