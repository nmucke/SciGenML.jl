
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

include("architectures.jl")
export DenseNeuralNetworkHyperparameters
export UNetHyperparameters

include("training.jl")
export TrainingHyperparameters
export OptimizerHyperparameters

include("models.jl")
export StochasticInterpolantHyperparameters
export FollmerStochasticInterpolantHyperparameters
export FlowMatchingHyperparameters
export ConditionalFlowMatchingHyperparameters
export ScoreBasedDiffusionModelHyperparameters

include("data.jl")
export KolmogorovDataHyperparameters
export SuperResKolmogorovDataHyperparameters
export KNMIDataHyperparameters

include("hyperparameters.jl")
export Hyperparameters

end
