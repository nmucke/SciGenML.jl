"""
    UNet(
        in_channels::Int,
        out_channels::Int,
        hidden_channels,
        time_embedding_dim::Int
        padding::String,
    )

    A UNet model.

    # Arguments
    - `in_channels::Int`: The number of input channels.
    - `out_channels::Int`: The number of output channels.
    - `hidden_channels::List{Int}`: The number of hidden channels in each layer.
    - `time_embedding_dim::Int`: The dimension of the time embedding.
    - `padding::String`: The padding type.

"""
function UNet(
    in_channels::Int,
    out_channels::Int,
    hidden_channels,
    time_embedding_dim::Int,
    padding::String
)
    reverse_hidden_channels = reverse(hidden_channels)

    model = Lux.@compact(

        # Conditioning embedding
        time_embedding = Lux.Chain(
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

        # time embedding
        t = time_embedding(t)

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

"""
    UNet(
        in_channels::Int,
        out_channels::Int,
        hidden_channels,
        time_embedding_dim::Int
        padding::String,
    )

    A UNet model.

    # Arguments
    - `in_channels::Int`: The number of input channels.
    - `out_channels::Int`: The number of output channels.
    - `hidden_channels::List{Int}`: The number of hidden channels in each layer.
    - `time_embedding_dim::Int`: The dimension of the time embedding.
    - `scalar_in_conditioning_dim::Int`: The dimension of the scalar input conditioning.
    - `scalar_hidden_conditioning_dim::Int`: The dimension of the hidden scalar conditioning.
    - `padding::String`: The padding type.

"""
function UNet(
    in_channels::Int,
    out_channels::Int,
    hidden_channels,
    time_embedding_dim::Int,
    scalar_in_conditioning_dim::Int,
    scalar_hidden_conditioning_dim::Int,
    padding::String
)
    reverse_hidden_channels = reverse(hidden_channels)

    conditioning_and_time_embedding_dim =
        time_embedding_dim + scalar_hidden_conditioning_dim

    model = Lux.@compact(

        # Time embedding
        time_embedding = Lux.Chain(
            x -> Layers.sinusoidal_embedding(x, time_embedding_dim),
            Lux.Dense(time_embedding_dim, time_embedding_dim; use_bias = true),
            DEFAULT_ACTIVATION_FUNCTION,
            Lux.Dense(time_embedding_dim, time_embedding_dim; use_bias = false)
        ),

        # Scalar conditioning embedding
        scalar_conditioning_embedding = Lux.Chain(
            Lux.Dense(
                scalar_in_conditioning_dim,
                scalar_hidden_conditioning_dim;
                use_bias = true
            ),
            DEFAULT_ACTIVATION_FUNCTION,
            Lux.Dense(
                scalar_hidden_conditioning_dim,
                scalar_hidden_conditioning_dim;
                use_bias = false
            )
        ),

        # Input
        input_conv = Lux.Conv((1, 1), (in_channels => hidden_channels[1])),

        ### Downsampling ###
        # conv next blocks
        down_conv_next_blocks = [
            Layers.multiple_conv_next_blocks(
                in_channels = hidden_channels[i],
                out_channels = hidden_channels[i + 1],
                conditioning_dim = conditioning_and_time_embedding_dim
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
            conditioning_dim = conditioning_and_time_embedding_dim
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
                conditioning_dim = conditioning_and_time_embedding_dim
            ) for i in 1:(length(reverse_hidden_channels) - 1)
        ],

        # Output
        output_conv = Lux.Conv((1, 1), (hidden_channels[1] => out_channels))
    ) do x
        x, c, t = x

        # input
        x = input_conv(x)

        # time embedding
        t = time_embedding(t)

        # conditioning embedding
        c = scalar_conditioning_embedding(c)
        c_and_t = vcat(c, t)

        # downsampling
        h = (x,)
        for (conv_next_block, down_sampling_layer) in
            zip(down_conv_next_blocks, down_sampling_layers)
            x = conv_next_block((x, c_and_t))
            h = (h..., x)
            x = down_sampling_layer(x)
        end

        # bottleneck
        x = bottleneck_conv_next_block((x, c_and_t))

        # upsampling
        for i in 1:length(up_conv_next_blocks)
            x = up_sampling_layers[i](x)
            x = h[end - i + 1] .+ x
            x = up_conv_next_blocks[i]((x, c_and_t))
        end

        # output
        x = h[1] .+ x
        x = output_conv(x)

        @return x
    end

    return model
end

"""
    UNet(
        in_channels::Int,
        out_channels::Int,
        hidden_channels,
        time_embedding_dim::Int
        padding::String,
    )

    A UNet model.

    # Arguments
    - `in_channels::Int`: The number of input channels.
    - `out_channels::Int`: The number of output channels.
    - `hidden_channels::List{Int}`: The number of hidden channels in each layer.
    - `time_embedding_dim::Int`: The dimension of the time embedding.
    - `padding::String`: The padding type.
    - `field_in_conditioning_dim::Int`: The dimension of the field input conditioning.
    - `field_hidden_conditioning_dim::Int`: The dimension of the hidden field conditioning.
    - `field_conditioning_combination::String`: The combination method for the field conditioning.

"""
function UNet(
    in_channels::Int,
    out_channels::Int,
    hidden_channels,
    time_embedding_dim::Int,
    padding::String,
    field_in_conditioning_dim::Int,
    field_hidden_conditioning_dim::Int,
    field_conditioning_combination::String
)
    reverse_hidden_channels = reverse(hidden_channels)

    initial_channels = in_channels + field_in_conditioning_dim

    if field_conditioning_combination == "concat"
        initial_channels = in_channels + field_hidden_conditioning_dim

        combine_field_conditioning_fn = (x, f) -> cat(x, f, dims = 3)
    elseif field_conditioning_combination == "add"
        initial_channels = in_channels

        combine_field_conditioning_fn = (x, f) -> x .+ f
    else
        throw(ArgumentError("Invalid field conditioning combination: $field_conditioning_combination"))
    end

    model = Lux.@compact(

        # Time embedding
        time_embedding = Lux.Chain(
            x -> Layers.sinusoidal_embedding(x, time_embedding_dim),
            Lux.Dense(time_embedding_dim, time_embedding_dim; use_bias = true),
            DEFAULT_ACTIVATION_FUNCTION,
            Lux.Dense(time_embedding_dim, time_embedding_dim; use_bias = false)
        ),

        # Field conditioning embedding
        field_conditioning_embedding = Lux.Conv(
            (1, 1),
            (field_in_conditioning_dim => field_hidden_conditioning_dim)
        ),
        combine_field_conditioning = combine_field_conditioning_fn,

        # Input
        input_conv = Lux.Conv((1, 1), (initial_channels => hidden_channels[1])),

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
        x, f, t = x

        # time embedding
        t = time_embedding(t)

        # conditioning embedding
        f = field_conditioning_embedding(f)

        # input
        x = combine_field_conditioning(x, f)
        x = input_conv(x)

        # # downsampling
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

"""
    UNet(config::Config.UNetHyperparameters)

    A UNet model.

    # Arguments
    - `config::Config.UNetHyperparameters`: The configuration for the UNet model.
"""
function UNet(config::Config.UNetHyperparameters)
    has_scalar_conditioning =
        !isnothing(config.scalar_in_conditioning_dim) &&
        !isnothing(config.scalar_hidden_conditioning_dim)
    has_field_conditioning =
        !isnothing(config.field_in_conditioning_dim) &&
        !isnothing(config.field_hidden_conditioning_dim)

    if !has_scalar_conditioning && !has_field_conditioning
        return UNet(
            config.in_channels,
            config.out_channels,
            config.hidden_channels,
            config.time_embedding_dim,
            config.padding
        )
    elseif has_scalar_conditioning && !has_field_conditioning
        return UNet(
            config.in_channels,
            config.out_channels,
            config.hidden_channels,
            config.time_embedding_dim,
            config.padding,
            config.scalar_in_conditioning_dim,
            config.scalar_hidden_conditioning_dim
        )
    elseif !has_scalar_conditioning && has_field_conditioning
        return UNet(
            config.in_channels,
            config.out_channels,
            config.hidden_channels,
            config.time_embedding_dim,
            config.padding,
            config.field_in_conditioning_dim,
            config.field_hidden_conditioning_dim,
            config.field_conditioning_combination
        )
    else
        throw(ArgumentError("Invalid combination of conditioning dimensions."))
    end
end
