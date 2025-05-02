
"""
    Config

Module for defining configuration options.

This module contains structs for storing configuration options. Most of the
structs are setup in a way that is similar to how one would do it in 
Python pydantic classes.
"""

module Config

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import Configurations

include("hyperparameters.jl")

export Hyperparameters
export DenseNeuralNetworkHyperparameters
export TrainingHyperparameters
export OptimizerHyperparameters
export StochasticInterpolantHyperparameters

end
