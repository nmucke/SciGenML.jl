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

export DEFAULT_TYPE

include("models/Models.jl")

include("architectures/Architectures.jl")

include("layers/Layers.jl")

include("time_integrators/TimeIntegrators.jl")

include("data/Data.jl")

include("sampling/Sampling.jl")

include("training/Training.jl")

include("config/Config.jl")

end
