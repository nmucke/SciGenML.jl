"""
    get_model(
        in_features,
        out_features,
        hidden_features,
        activation_function = DEFAULT_ACTIVATION_FUNCTION,
        dropout = DEFAULT_DROPOUT
    )

    in_features: The number of input features.
    out_features: The number of output features.
    hidden_features: The number of hidden features.
    activation_function: The activation function.
    dropout: The dropout rate.

    Returns a model that can be used to predict the output from the input.
"""
function get_model(;
    in_features,
    out_features,
    hidden_features,
    activation_function = DEFAULT_ACTIVATION_FUNCTION,
    dropout = DEFAULT_DROPOUT
)
    model = Lux.@compact(
        input_layer = Lux.Dense(in_features, hidden_features[1]; use_bias = true),
        batch_norm_layers = [
            Lux.BatchNorm(hidden_features[i]) for i in 1:(length(hidden_features) - 1)
        ],
        dropout_layers =
            [Lux.Dropout(dropout) for i in 1:(length(hidden_features) - 1)],
        hidden_layers = [
            Lux.Dense(hidden_features[i], hidden_features[i + 1]; use_bias = true)
            for i in 1:(length(hidden_features) - 1)
        ],
        output_layer = Lux.Dense(hidden_features[end], out_features; use_bias = false),
        activation_function = activation_function
    ) do x
        x = activation_function(input_layer(x))
        for i in eachindex(hidden_layers)
            x = batch_norm_layers[i](x)
            x = dropout_layers[i](x)
            x = hidden_layers[i](x)
            x = activation_function(x)
        end
        @return output_layer(x)
    end

    return model
end

"""
    DenseNN(
        in_features, 
        out_features, 
        hidden_features;
        activation_function = DEFAULT_ACTIVATION_FUNCTION,
        dropout = DEFAULT_DROPOUT
    )

    in_features: The number of input features.
    out_features: The number of output features.
    hidden_features: The number of hidden features.
    activation_function: The activation function.

A dense neural network with `in_features` input features, 
`out_features` output features, and `hidden_features` hidden features.
"""

function DenseNeuralNetwork(
    in_features::Int,
    out_features,
    hidden_features;
    activation_function = DEFAULT_ACTIVATION_FUNCTION,
    dropout = DEFAULT_DROPOUT
)
    return get_model(;
        in_features = in_features,
        out_features = out_features,
        hidden_features = hidden_features,
        activation_function = activation_function,
        dropout = dropout
    )
end

"""
    DenseNN(
        in_features::Tuple,
        out_features::Int, 
        hidden_features::Vector{Int};
        activation_function::Function = DEFAULT_ACTIVATION_FUNCTION,
        dropout::Float32 = DEFAULT_DROPOUT
    )

    in_features: A tuple of the number of input features
    out_features: The number of output features.
    hidden_features: The number of hidden features.
    activation_function: The activation function.

    A dense neural network with `in_features` input features, 
    `out_features` output features, and `hidden_features` hidden features.
    The input is a tuple that are concatenated to form a single input.
"""
function DenseNeuralNetwork(
    in_features::Tuple,
    out_features::Int,
    hidden_features::Vector{Int};
    activation_function::Function = DEFAULT_ACTIVATION_FUNCTION,
    dropout::Float32 = DEFAULT_DROPOUT
)
    model = Lux.@compact(network = get_model(;
        in_features = sum(in_features),
        out_features = out_features,
        hidden_features = hidden_features,
        activation_function = activation_function,
        dropout = dropout
    )) do x
        x = vcat(x...)
        @return network(x)
    end

    return model
end
