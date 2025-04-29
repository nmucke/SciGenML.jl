
"""
    DenseNN(
        in_features::Int, 
        out_features::Int, 
        hidden_features::Vector{Int}, 
        activation_function::Function = identity
    )

    in_features: The number of input features.
    out_features: The number of output features.
    hidden_features: The number of hidden features.
    activation_function: The activation function.

A dense neural network with `in_features` input features, 
`out_features` output features, and `hidden_features` hidden features.
"""

function DenseNN(;
        in_features::Int,
        out_features::Int,
        hidden_features::Vector{Int},
        activation_function::Function = identity
)
    model = Lux.@compact(input_layer=Lux.Dense(
            in_features, hidden_features[1]; use_bias = true),
        hidden_layers=[Lux.Dense(
                           hidden_features[i], hidden_features[i + 1]; use_bias = true)
                       for
                       i in 1:(length(hidden_features) - 1)],
        output_layer=Lux.Dense(hidden_features[end], out_features; use_bias = false),
        activation_function=activation_function) do x
        x = activation_function(input_layer(x))
        for layer in hidden_layers
            x = activation_function(layer(x))
        end
        x = output_layer(x)
        @return x
    end

    return model
end
