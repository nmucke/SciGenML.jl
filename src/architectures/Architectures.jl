"""
    Architectures

Module for defining neural network architectures.

This module contains full neural network architectures/models. In contrast
to the `Layers` module, which contains individual layers, this module contains
complete models.
"""

module Architectures

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import Lux

const DEFAULT_ACTIVATION_FUNCTION = x -> Lux.leakyrelu(x, 0.1)
const DEFAULT_DROPOUT = 0.1 |> DEFAULT_TYPE

include("dense.jl")

export DenseNeuralNetwork

end
