module Config

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import Configurations

include("hyperparameters.jl")

export Hyperparameters
export DenseNeuralNetworkHyperparameters
export TrainingHyperparameters
export OptimizerHyperparameters

end
