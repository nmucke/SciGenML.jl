module SciGenML

include("models/Models.jl")

include("neural_network_architectures/NeuralNetworkArchitectures.jl")

include("neural_network_layers/NeuralNetworkLayers.jl")

include("time_integrators/TimeIntegrators.jl")

include("data/Data.jl")

include("sampling/Sampling.jl")
end
