module Models

import Lux

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

export GenerativeModel, ConditionalGenerativeModel

"""
Abstract type for all generative models.
"""
abstract type GenerativeModel end

"""
Abstract type for all conditional generative models.
"""
abstract type ConditionalGenerativeModel <: GenerativeModel end

##### Stochastic Interpolants #####
include("stochastic_interpolants/interpolants.jl")
include("stochastic_interpolants/generative_model.jl")
include("stochastic_interpolants/utils.jl")

### Interpolants
export InterpolantCoefs

# Specific interpolants
export linear_interpolant_coefs, quadratic_interpolant_coefs

# Interpolant functions
export compute_interpolant, compute_interpolant_diff

# Stochastic interpolant generative model
export StochasticInterpolantGenerativeModel

##### Models #####

end
