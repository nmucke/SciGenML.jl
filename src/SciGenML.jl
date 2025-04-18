module SciGenML

include("models/Models.jl")
export Models

include("neural_network_architectures/NeuralNetworkArchitectures.jl")
export NeuralNetworkArchitectures

include("neural_network_layers/NeuralNetworkLayers.jl")
export NeuralNetworkLayers

include("time_integrators/TimeIntegrators.jl")
# export TimeIntegrators

include("data/Data.jl")
export Data

end
