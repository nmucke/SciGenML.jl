"""
    UNet(
        in_channels::Int,
        out_channels::Int,
        hidden_channels::List{Int},
        padding::String,
    )

    A UNet model.

    # Arguments
    - `in_channels::Int`: The number of input channels.
    - `out_channels::Int`: The number of output channels.
    - `hidden_channels::List{Int}`: The number of hidden channels in each layer.
    - `padding::String`: The padding type.

"""
function UNet(
    in_channels::Int,
    out_channels::Int,
    hidden_channels,
    in_conditioning_dim::Int,
    time_embedding_dim::Int,
    padding::String
)
    reverse_hidden_channels = reverse(hidden_channels)

    model = Lux.@compact(

        # Conditioning embedding
        conditioning_embedding = Lux.Chain(
            x -> Layers.sinusoidal_embedding(x, time_embedding_dim),
            Lux.Dense(time_embedding_dim, time_embedding_dim; use_bias = true),
            DEFAULT_ACTIVATION_FUNCTION,
            Lux.Dense(time_embedding_dim, time_embedding_dim; use_bias = true)
        ),

        # Input
        input_conv = Lux.Conv((1, 1), (in_channels => hidden_channels[1])),

        ### Downsampling ###
        # conv next blocks
        down_conv_next_blocks = [
            Layers.multiple_conv_next_blocks(
                in_channels = hidden_channels[i],
                out_channels = hidden_channels[i + 1],
                conditioning_dim = time_embedding_dim
            ) for i in 1:(length(hidden_channels) - 1)
        ],

        # downsampling layers
        down_sampling_layers = [
            Lux.Chain(
                Layers.get_padding(padding, 1),
                Lux.Conv(
                    (4, 4),
                    (hidden_channels[i + 1] => hidden_channels[i + 1]);
                    stride = (2, 2)
                )
            ) for i in 1:(length(hidden_channels) - 1)
        ],

        ### Bottleneck ###
        bottleneck_conv_next_block = Layers.multiple_conv_next_blocks(
            in_channels = hidden_channels[end],
            out_channels = hidden_channels[end],
            conditioning_dim = time_embedding_dim
        ),

        ### Upsampling ###
        # upsampling layers
        up_sampling_layers = [
            Lux.ConvTranspose(
                (4, 4),
                (reverse_hidden_channels[i] => reverse_hidden_channels[i]);
                stride = (2, 2),
                pad = 1
            ) for i in 1:(length(reverse_hidden_channels) - 1)
        ],

        # conv next blocks
        up_conv_next_blocks = [
            Layers.conv_next_block(
                in_channels = reverse_hidden_channels[i],
                out_channels = reverse_hidden_channels[i + 1],
                conditioning_dim = time_embedding_dim
            ) for i in 1:(length(reverse_hidden_channels) - 1)
        ],

        # Output
        output_conv = Lux.Conv((1, 1), (hidden_channels[1] => out_channels))
    ) do x
        x, t = x

        # input
        x = input_conv(x)

        # conditioning embedding
        t = conditioning_embedding(t)

        # downsampling
        h = (x,)
        for (conv_next_block, down_sampling_layer) in
            zip(down_conv_next_blocks, down_sampling_layers)
            x = conv_next_block((x, t))
            h = (h..., x)
            x = down_sampling_layer(x)
        end

        # bottleneck
        x = bottleneck_conv_next_block((x, t))

        # upsampling
        for i in 1:length(up_conv_next_blocks)
            x = up_sampling_layers[i](x)
            x = h[end - i + 1] .+ x
            x = up_conv_next_blocks[i]((x, t))
        end

        # output
        x = h[1] .+ x
        x = output_conv(x)

        @return x
    end

    return model
end
