
###############################################################################
# Conv Next Block
###############################################################################

"""
    conv_next_block_no_conditioning(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        imsize::Tuple{Int, Int} = (64, 128)
    )

Create a conv next block with the given number of input and output channels. 
The block consists of two convolutional layers with kernel size `kernel_size`.
The first layer has the same number of input and output channels, while the 
second layer has the same number of output channels as the block. 
The block also includes batch normalization and a skip connection.

https://arxiv.org/abs/2201.03545

Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/b9b33913b99ede88d9452c5ab470c5d7f5da5c56/src/turbpred/model_diffusion_blocks.py#L60

"""
function conv_next_block_no_conditioning(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    padding = "constant"
)
    model = Lux.@compact(
        ds_conv = Chain(
            get_padding(padding, 3),
            Lux.Conv((7, 7), in_channels => in_channels)#, groups=in_channels)
        ),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels => in_channels * multiplier); pad = 0),
            NNlib.gelu,
            Lux.InstanceNorm(in_channels * multiplier),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels * multiplier => out_channels); pad = 0)
        ),
        res_conv = Lux.Conv((1, 1), (in_channels => out_channels); pad = 0)
    ) do x
        h = ds_conv(x)
        h = conv_net(h)

        @return h .+ res_conv(x)
    end

    return model
end

"""
    multiple_conv_next_blocks_no_conditioning(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        padding="constant"
    )

Create a chain of two conv next blocks with the given number of input and
output channels. The first block has the same number of input and output

"""
function multiple_conv_next_blocks_no_conditioning(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 1,
    num_blocks::Int = 2,
    padding = "constant"
)
    model = Lux.@compact(blocks = [
        conv_next_block_no_conditioning(
            in_channels = in_channels,
            out_channels = out_channels,
            multiplier = multiplier,
            padding = padding
        ) for _ in 1:num_blocks
    ],) do x
        for block in blocks
            x = block(x)
        end
        @return x
    end

    return model
end

"""
    conv_next_block(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        pars_embed_dim::Int = 1,
        imsize::Tuple{Int, Int} = (64, 128)
    )

Create a conv next block with the given number of input and output channels. 
The block consists of two convolutional layers with kernel size `kernel_size`.
The first layer has the same number of input and output channels, while the 
second layer has the same number of output channels as the block. 
The block also includes batch normalization and a skip connection.

https://arxiv.org/abs/2201.03545

Based on https://github.com/tum-pbs/autoreg-pde-diffusion/blob/b9b33913b99ede88d9452c5ab470c5d7f5da5c56/src/turbpred/model_diffusion_blocks.py#L60

"""
function conv_next_block(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 2,
    conditioning_dim::Int = 1,
    padding = "constant"
)
    model = Lux.@compact(
        ds_conv = Chain(
            get_padding(padding, 3),
            Lux.Conv((7, 7), in_channels => in_channels)#, groups=in_channels)
        ),
        pars_mlp = Chain(Lux.Dense(conditioning_dim => in_channels)),
        conv_net = Chain(
            Lux.InstanceNorm(in_channels),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels => in_channels * multiplier); pad = 0),
            NNlib.gelu,
            Lux.InstanceNorm(in_channels * multiplier),
            get_padding(padding, 1),
            Lux.Conv((3, 3), (in_channels * multiplier => out_channels); pad = 0)
        ),
        res_conv = Lux.Conv((1, 1), (in_channels => out_channels); pad = 0)
    ) do x
        x, c = x
        h = ds_conv(x)
        c = pars_mlp(c)
        c = reshape(c, 1, 1, size(c)...)
        h = h .+ c
        h = conv_net(h)

        @return h .+ res_conv(x)
    end

    return model
end

"""
    multiple_conv_next_blocks(
        in_channels::Int, 
        out_channels::Int,
        multiplier::Int = 1,
        conditioning_dim::Int = 1,
        imsize::Tuple{Int, Int} = (64, 128)
    )

Create a chain of two conv next blocks with the given number of input and
output channels. The first block has the same number of input and output
"""
function multiple_conv_next_blocks(;
    in_channels::Int,
    out_channels::Int,
    multiplier::Int = 2,
    conditioning_dim::Int = 1,
    num_blocks::Int = 2,
    padding = "constant"
)
    if num_blocks == 1
        channels = [in_channels, out_channels]
    else
        channels = [in_channels for _ in 1:num_blocks]
        channels = [channels..., out_channels]
    end

    model = Lux.@compact(blocks = [
        conv_next_block(
            in_channels = channels[i],
            out_channels = channels[i + 1],
            multiplier = multiplier,
            conditioning_dim = conditioning_dim,
            padding = padding
        ) for i in 1:num_blocks
    ],) do x
        x, c = x
        for block in blocks
            x = block((x, c))
        end
        @return x
    end

    return model
end
