"""
    Models

Module for defining models.

This module contains generative models.
"""

module Models

import Lux

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.ZERO_TOL as ZERO_TOL
import SciGenML.DEFAULT_DEVICE as DEFAULT_DEVICE

import SciGenML.Config as Config
import SciGenML.Architectures as Architectures
import SciGenML.Utils as Utils
export GenerativeModel, ConditionalGenerativeModel

"""
Abstract type for all generative models.
"""
abstract type GenerativeModel end

"""
Abstract type for all conditional generative models.
"""
abstract type ConditionalGenerativeModel <: GenerativeModel end

##### GenerativeTraits #####
struct Deterministic end
struct Stochastic end

export Deterministic, Stochastic

##### Stochastic Interpolants #####
include("stochastic_interpolants/interpolants.jl")
include("stochastic_interpolants/generative_model.jl")
include("stochastic_interpolants/stochastic_interpolant_utils.jl")
include("stochastic_interpolants/follmer_generative_model.jl")

##### Flow Matching #####
include("flow_matching/generative_model.jl")
include("flow_matching/conditional_generative_model.jl")
### Interpolants
export DeterministicInterpolantCoefs, StochasticInterpolantCoefs

# Specific interpolants
export linear_interpolant_coefs, quadratic_interpolant_coefs, diffusion_interpolant_coefs

# Interpolant functions
export compute_interpolant, compute_interpolant_diff

# Stochastic interpolant generative model
export StochasticInterpolant

# Follmer stochastic interpolant generative model
export FollmerStochasticInterpolant

# drift term
export drift_term

# Flow matching generative model
export FlowMatching

# Conditional flow matching generative model
export ConditionalFlowMatching

# Score-based diffusion model
include("denoising_diffusion_model/generative_model.jl")

export ScoreBasedDiffusionModel

end
