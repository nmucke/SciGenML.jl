"""
    SciGenML

Module for the SciGenML package.

This module is the main module for the SciGenML package. It imports all the
other modules in the package and makes them available to the user.
"""

module SciGenML

using LuxCUDA
import Lux
import Optimisers
using Zygote

DEFAULT_TYPE = Float32
ZERO_TOL = 1.0f-12

DEFAULT_DEVICE = Lux.gpu_device()

export DEFAULT_TYPE, ZERO_TOL

include("config/Config.jl")

include("layers/Layers.jl")

include("architectures/Architectures.jl")

include("utils/Utils.jl")

include("models/Models.jl")

include("time_integrators/TimeIntegrators.jl")

include("data/Data.jl")

include("sampling/Sampling.jl")

include("training/Training.jl")

include("plotting/Plotting.jl")

end
