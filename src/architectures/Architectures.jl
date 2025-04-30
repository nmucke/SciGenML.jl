module Architectures

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import Lux

const DEFAULT_ACTIVATION_FUNCTION = x -> Lux.leakyrelu(x, 0.1)
const DEFAULT_DROPOUT = 0.1 |> DEFAULT_TYPE

include("dense.jl")

export DenseNeuralNetwork

end
