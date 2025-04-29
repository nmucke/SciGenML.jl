module Training

import SciGenML.DEFAULT_TYPE as DEFAULT_TYPE

import Lux
import Random
import Optimisers
import ProgressBars
import Zygote

DEFAULT_LR = DEFAULT_TYPE(1.0f-3)
DEFAULT_LAMBDA = DEFAULT_TYPE(1.0f-3)
DEFAULT_OPTIMIZER = Optimisers.AdamW(; eta = DEFAULT_LR, lambda = DEFAULT_LAMBDA)
DEFAULT_LOSS_FN = Lux.MSELoss()
DEFAULT_NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16

export DEFAULT_LR, DEFAULT_LAMBDA, DEFAULT_OPTIMIZER, DEFAULT_LOSS_FN, DEFAULT_NUM_EPOCHS

include("simple_train.jl")

export simple_train

end
