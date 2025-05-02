"""
    SciGenML

Module for the SciGenML package.

This module is the main module for the SciGenML package. It imports all the
other modules in the package and makes them available to the user.
"""

module SciGenML

import Lux
import Optimisers
using Zygote

DEFAULT_TYPE = Float32
ZERO_TOL = 1.0f-12

export DEFAULT_TYPE, ZERO_TOL

include("utils/Utils.jl")

include("config/Config.jl")

include("architectures/Architectures.jl")

include("models/Models.jl")

include("layers/Layers.jl")

include("time_integrators/TimeIntegrators.jl")

include("data/Data.jl")

include("sampling/Sampling.jl")

include("training/Training.jl")

end
