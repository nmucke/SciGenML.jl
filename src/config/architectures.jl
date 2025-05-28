
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
    scalar_in_conditioning_dim::Union{Int, Nothing} = nothing
    scalar_hidden_conditioning_dim::Union{Int, Nothing} = nothing
    field_in_conditioning_dim::Union{Int, Nothing} = nothing
    field_hidden_conditioning_dim::Union{Int, Nothing} = nothing
    field_conditioning_combination::Union{String, Nothing} = nothing

    # CONSTRUCTOR
    function UNetHyperparameters(
        in_channels,
        out_channels,
        hidden_channels,
        time_embedding_dim,
        padding,
        scalar_in_conditioning_dim::Union{Int, Nothing} = nothing,
        scalar_hidden_conditioning_dim::Union{Int, Nothing} = nothing,
        field_in_conditioning_dim::Union{Int, Nothing} = nothing,
        field_hidden_conditioning_dim::Union{Int, Nothing} = nothing,
        field_conditioning_combination::Union{String, Nothing} = nothing
    )
        has_scalar_conditioning =
            !isnothing(scalar_in_conditioning_dim) &&
            !isnothing(scalar_hidden_conditioning_dim)
        has_field_conditioning =
            !isnothing(field_in_conditioning_dim) &&
            !isnothing(field_hidden_conditioning_dim)

        if !has_scalar_conditioning && !has_field_conditioning
            return new(
                in_channels,
                out_channels,
                hidden_channels,
                time_embedding_dim,
                padding
            )
        elseif has_scalar_conditioning && !has_field_conditioning
            return new(
                in_channels,
                out_channels,
                hidden_channels,
                time_embedding_dim,
                padding,
                scalar_in_conditioning_dim,
                scalar_hidden_conditioning_dim
            )
        elseif !has_scalar_conditioning && has_field_conditioning
            return new(
                in_channels,
                out_channels,
                hidden_channels,
                time_embedding_dim,
                padding,
                nothing,
                nothing,
                field_in_conditioning_dim,
                field_hidden_conditioning_dim,
                field_conditioning_combination
            )
        else
            throw(ArgumentError("Invalid combination of conditioning dimensions."))
        end
    end
end
