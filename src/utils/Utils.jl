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

export get_iter

end
