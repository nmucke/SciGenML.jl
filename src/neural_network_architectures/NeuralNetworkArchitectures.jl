module NeuralNetworkArchitectures

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

using Lux: Lux

const DEFAULT_ACTIVATION_FUNCTION = x -> Lux.relu(x)
const DEFAULT_DROPOUT = 0.1 |> DEFAULT_TYPE

include("dense.jl")

export DenseNeuralNetwork

end
