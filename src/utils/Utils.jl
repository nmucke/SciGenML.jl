"""
    Utils

Module for utility functions.
"""

module Utils

import ProgressBars

function get_iter(num_epochs::Int, verbose::Bool)
    if verbose
        return ProgressBars.ProgressBar(1:num_epochs)
    else
        return 1:num_epochs
    end
end

function move_to_device(model, device)
    model.ps = model.ps |> device

    if model.st !== nothing
        model.st = model.st |> device
    end

    model.device = device
    return model
end

reshape_scalar(x, ndims) = reshape(x, ntuple(i -> i == ndims ? size(x)[end] : 1, ndims))

export get_iter, move_to_device, reshape_scalar

end
