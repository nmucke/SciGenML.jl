
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
    UNetHyperparameters

    Hyperparameters for the UNet.
"""
Configurations.@option "u_net" struct UNetHyperparameters
    in_channels::Int
    out_channels::Int
    hidden_channels::Vector{Int}
    time_embedding_dim::Int
    padding::String
    in_conditioning_dim::Union{Int, Nothing} = nothing
    hidden_conditioning_dim::Union{Int, Nothing} = nothing

    # CONSTRUCTOR
    function UNetHyperparameters(
        in_channels,
        out_channels,
        hidden_channels,
        time_embedding_dim,
        padding,
        scalar_in_conditioning_dim::Union{Int, Nothing} = nothing,
        scalar_hidden_conditioning_dim::Union{Int, Nothing} = nothing
    )
        if isnothing(scalar_in_conditioning_dim) &&
           isnothing(scalar_hidden_conditioning_dim)
            return new(
                in_channels,
                out_channels,
                hidden_channels,
                time_embedding_dim,
                padding
            )
        else
            return new(
                in_channels,
                out_channels,
                hidden_channels,
                time_embedding_dim,
                padding,
                scalar_in_conditioning_dim,
                scalar_hidden_conditioning_dim
            )
        end
    end
end

"""
    TrainingHyperparameters

    Hyperparameters for the training.
"""
Configurations.@option struct TrainingHyperparameters
    batch_size::Int
    num_epochs::Int
    match_base_and_target::Bool
end

"""
    OptimizerHyperparameters

    Hyperparameters for the optimizer.
"""
Configurations.@option struct OptimizerHyperparameters
    type::String
    learning_rate::DEFAULT_TYPE
    weight_decay::DEFAULT_TYPE

    # CONSTRUCTOR
    function OptimizerHyperparameters(type, learning_rate, weight_decay)
        return new(type, learning_rate |> DEFAULT_TYPE, weight_decay |> DEFAULT_TYPE)
    end
end

"""
    FlowMatchingHyperparameters

    Hyperparameters for the flow matching generative model.
"""
Configurations.@option "flow_matching" struct FlowMatchingHyperparameters
    interpolant_type::String
end

"""
    ConditionalFlowMatchingHyperparameters

    Hyperparameters for the conditional flow matching generative model.
"""
Configurations.@option "conditional_flow_matching" struct ConditionalFlowMatchingHyperparameters
    interpolant_type::String
    guidance_scale::DEFAULT_TYPE
    replacement_probability::DEFAULT_TYPE
    unconditional_condition::DEFAULT_TYPE
end

"""
    StochasticInterpolantHyperparameters

    Hyperparameters for the stochastic interpolant generative model.
"""
Configurations.@option "stochastic_interpolant" struct StochasticInterpolantHyperparameters
    interpolant_type::String
end

"""
    ScoreBasedDiffusionModelHyperparameters

    Hyperparameters for the score-based diffusion model generative model.
"""
Configurations.@option "score_based_diffusion_model" struct ScoreBasedDiffusionModelHyperparameters
    interpolant_type::String
end

"""
    Hyperparameters

    Hyperparameters for the architecture, training, and optimizer.
"""
Configurations.@option struct Hyperparameters
    architecture::Union{DenseNeuralNetworkHyperparameters, UNetHyperparameters}
    training::TrainingHyperparameters
    optimizer::OptimizerHyperparameters
    model::Union{
        StochasticInterpolantHyperparameters,
        FlowMatchingHyperparameters,
        ConditionalFlowMatchingHyperparameters,
        ScoreBasedDiffusionModelHyperparameters
    }
end
