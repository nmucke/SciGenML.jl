"""
    Utils

Module for utility functions.
"""

module Utils

import ProgressBars
import SciGenML.Models as Models

function get_iter(num_epochs::Int, verbose::Bool)
    if verbose
        return ProgressBars.ProgressBar(1:num_epochs)
    else
        return 1:num_epochs
    end
end

function move_to_device(model::Models.GenerativeModel, device)
    model.ps = model.ps |> device
    model.st = model.st |> device
    model.device = device
    return model
end

export get_iter, move_to_device

end
