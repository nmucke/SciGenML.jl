"""
    Training

Module for training models.

This module contains functions for training generative models.
"""

module Training

import SciGenML
import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.ZERO_TOL as ZERO_TOL
import SciGenML.DEFAULT_DEVICE as DEFAULT_DEVICE

import SciGenML.Models as Models
import SciGenML.Utils as Utils
import SciGenML.Sampling as Sampling

import Lux
import Random
import Optimisers
import ProgressBars
import Zygote
import Distributions
import Statistics
import DataLoaders

const DEFAULT_LR = DEFAULT_TYPE(1.0f-3)
const DEFAULT_LAMBDA = DEFAULT_TYPE(1.0f-3)
const DEFAULT_OPTIMIZER = Optimisers.AdamW(; eta = DEFAULT_LR, lambda = DEFAULT_LAMBDA)
const DEFAULT_LOSS_FN = Lux.MSELoss()
const DEFAULT_NUM_EPOCHS = 100
const DEFAULT_BATCH_SIZE = 16

const MSE_LOSS_FN = Lux.MSELoss()

export DEFAULT_LR,
    DEFAULT_LAMBDA,
    DEFAULT_OPTIMIZER,
    DEFAULT_LOSS_FN,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_BATCH_SIZE,
    MSE_LOSS_FN

"""
    train(
        model,
        args...; 
        kwargs...
    )

    Generic training function for all models.
"""
function train(model, args...; kwargs...)
    return train(model.trait, model, args...; kwargs...)
end

##### Training Utils #####

include("training_utils.jl")

export get_optimizer, get_dataloader, get_interpolated_samples, get_gradients

##### Simple Training #####

include("simple_train.jl")

export simple_train

##### Stochastic Interpolant Training #####
include("train_stochastic_interpolant.jl")

##### Flow Matching Training #####
include("train_flow_matching.jl")

##### Diffusion Model Training #####
include("train_diffusion_model.jl")

export train

end
