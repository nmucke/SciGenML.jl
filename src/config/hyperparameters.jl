
"""
    DenseNeuralNetworkHyperparameters

    Hyperparameters for the dense neural network.
"""
Configurations.@option "dense_neural_network" struct DenseNeuralNetworkHyperparameters
    in_features::Any
    out_features::Int
    hidden_features::Vector{Int}
    dropout::Any

    # CONSTRUCTOR
    function DenseNeuralNetworkHyperparameters(
        in_features,
        out_features,
        hidden_features,
        dropout
    )
        if typeof(in_features) == Vector{Int}
            in_features = Tuple(in_features)
        end
        return new(in_features, out_features, hidden_features, dropout)
    end
end

"""
    PlaceholderHyperparameters

    Hyperparameters for the placeholder.
"""
Configurations.@option "placeholder" struct PlaceholderHyperparameters
    in_features::Any
    out_features::Int
    channels::Vector{Int}
    hidden_features::Vector{Int}
    dropout::Any

    # CONSTRUCTOR
    function PlaceholderHyperparameters(
        in_features,
        out_features,
        channels,
        hidden_features,
        dropout
    )
        if typeof(in_features) == Vector{Int}
            in_features = Tuple(in_features)
        end
        return new(in_features, out_features, channels, hidden_features, dropout)
    end
end

"""
    TrainingHyperparameters

    Hyperparameters for the training.
"""
Configurations.@option struct TrainingHyperparameters
    batch_size::Int
    num_epochs::Int
end

"""
    OptimizerHyperparameters

    Hyperparameters for the optimizer.
"""
Configurations.@option struct OptimizerHyperparameters
    learning_rate::DEFAULT_TYPE
    weight_decay::DEFAULT_TYPE

    # CONSTRUCTOR
    function OptimizerHyperparameters(learning_rate, weight_decay)
        return new(learning_rate |> DEFAULT_TYPE, weight_decay |> DEFAULT_TYPE)
    end
end

"""
    PlaceholderGenerativeModel

    A generative model that uses a placeholder.
"""
Configurations.@option "placeholder" struct PlaceholderGenerativeModel end

"""
    StochasticInterpolantGenerativeModelHyperparameters

    Hyperparameters for the stochastic interpolant generative model.
"""
Configurations.@option "stochastic_interpolant" struct StochasticInterpolantGenerativeModelHyperparameters
    interpolant_type::String
end

"""
    Hyperparameters

    Hyperparameters for the architecture, training, and optimizer.
"""
Configurations.@option struct Hyperparameters
    architecture::Union{DenseNeuralNetworkHyperparameters, PlaceholderHyperparameters}
    training::TrainingHyperparameters
    optimizer::OptimizerHyperparameters
    model::Union{
        StochasticInterpolantGenerativeModelHyperparameters,
        PlaceholderGenerativeModel
    }
end
