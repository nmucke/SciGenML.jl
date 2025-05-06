"""
    Architectures

Module for defining neural network architectures.

This module contains full neural network architectures/models. In contrast
to the `Layers` module, which contains individual layers, this module contains
complete models.
"""

module Architectures

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import SciGenML.Layers as Layers
import Lux
import NNlib

const DEFAULT_ACTIVATION_FUNCTION = NNlib.gelu
const DEFAULT_DROPOUT = 0.1 |> DEFAULT_TYPE

### Dense ###
include("dense.jl")

export DenseNeuralNetwork

### UNet ###
include("unet.jl")

export UNet

end
