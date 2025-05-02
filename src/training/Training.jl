"""
    Training

Module for training models.

This module contains functions for training generative models.
"""

module Training

import SciGenML
import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE
import SciGenML.Models as Models
import Lux
import Random
import Optimisers
import ProgressBars
import Zygote
import Distributions
import Statistics

DEFAULT_LR = DEFAULT_TYPE(1.0f-3)
DEFAULT_LAMBDA = DEFAULT_TYPE(1.0f-3)
DEFAULT_OPTIMIZER = Optimisers.AdamW(; eta = DEFAULT_LR, lambda = DEFAULT_LAMBDA)
DEFAULT_LOSS_FN = Lux.MSELoss()
DEFAULT_NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16

export DEFAULT_LR, DEFAULT_LAMBDA, DEFAULT_OPTIMIZER, DEFAULT_LOSS_FN, DEFAULT_NUM_EPOCHS

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

include("utils.jl")

export get_optimizer, prepare_batches

##### Simple Training #####

include("simple_train.jl")

export simple_train

##### Stochastic Interpolant Training #####

include("train_stochastic_interpolant.jl")

export train

end
