"""
    Sampling

Module for sampling from models.

This module contains functions for sampling using generative models.
"""

module Sampling

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.Models as Models

import Random
import Distributions
import ProgressBars
##### Stochastic Interpolant Sampling #####

include("sample_stochastic_interpolant.jl")

export sample

end
