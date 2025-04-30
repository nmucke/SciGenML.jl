module SciGenML

import Lux
import Optimisers
using Zygote

DEFAULT_TYPE = Float32

export DEFAULT_TYPE

include("models/Models.jl")

include("architectures/Architectures.jl")

include("neural_network_layers/NeuralNetworkLayers.jl")

include("time_integrators/TimeIntegrators.jl")

include("data/Data.jl")

include("sampling/Sampling.jl")

include("training/Training.jl")

include("config/Config.jl")

end
