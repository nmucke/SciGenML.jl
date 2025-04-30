module Models

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

### Interpolants
export InterpolantCoefs

# Specific interpolants
export linear_interpolant, quadratic_interpolant

# Interpolant functions
export compute_interpolant, compute_interpolant_diff

##### Models #####

end
