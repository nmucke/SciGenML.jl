"""
    Layers

Module for defining individual layers.

This module contains individual layers that can be used to build up larger
neural network architectures.
"""

module Layers

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.DEFAULT_DEVICE as DEFAULT_DEVICE

using Lux
using Random
using NNlib

### Layer Utils ###
include("layer_utils.jl")
export get_padding

### Embeddings ###
include("embeddings.jl")
export sinusoidal_embedding

### ConvNextLayers ###
include("conv_next_layers.jl")
export conv_next_block_no_conditioning, multiple_conv_next_blocks_no_conditioning
export conv_next_block, multiple_conv_next_blocks

end
