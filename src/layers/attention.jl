
"""
    _flatten_spatial(x::AbstractArray{T, 4})

Flattens the first 2 dimensions of `x`, and permutes the remaining dimensions to (2, 1, 3)
"""
@inline function _flatten_spatial(x::AbstractArray{T, 4}) where {T}
    return permutedims(reshape(x, (:, size(x, 3), size(x, 4))), (2, 1, 3))
end

"""
    patchify(
        imsize::Tuple{Int, Int}=(64, 64),
        in_channels::Int=3,
        patch_size::Tuple{Int, Int}=(8, 8),
        embed_planes::Int=128,
        norm_layer=Returns(Lux.NoOpLayer()),
        flatten=true
    )

Create a patch embedding layer with the given image size, number of input
channels, patch size, embedding planes, normalization layer, and flatten flag.

Based on https://github.com/LuxDL/Boltz.jl/blob/v0.3.9/src/vision/vit.jl#L48-L61
"""
function patchify(
    imsize::Tuple{Int, Int}; 
    in_channels=3, 
    patch_size=(8, 8),
    embed_planes=128, 
    norm_layer=Returns(Lux.NoOpLayer()), 
    flatten=true,
    with_embedding=false
)
    im_width, im_height = imsize
    patch_width, patch_height = patch_size

    # Calculate number of patches
    num_patches_h = im_height ÷ patch_height
    num_patches_w = im_width ÷ patch_width
    num_patches = num_patches_h * num_patches_w

    # Create model to extract patches
    model = Lux.@compact(
        proj = Chain(
            Lux.Conv((patch_width, patch_height), in_channels => embed_planes; stride=patch_size),
            norm_layer()
        )
    ) do x
        # Project patches
        h = proj(x)
        
        if flatten
            # Reshape to (embed_dim, num_patches, batch)
            h = _flatten_spatial(h)
            
            if with_embedding
                # Add learnable position embedding
                pos_embed = Lux.Embedding(num_patches => embed_planes)(1:num_patches)
                h = h .+ reshape(pos_embed, size(pos_embed, 1), size(pos_embed, 2), 1)
            end
        end
        
        @return h
    end

    return model


end

"""
    unpatchify(x, patch_size, out_channels)

Unpatchify the input tensor `x` with the given patch size and number of output
channels.
"""
function unpatchify(x, imsize, patch_size, out_channels)
    
    c = out_channels
    p1, p2 = patch_size
    h, w = imsize
    @assert h * w == size(x, 2)

    x = reshape(x, (p1, p2, c, h, w, size(x, 3)))
    x = permutedims(x, (1, 4, 2, 5, 3, 6))
    imgs = reshape(x, (h * p1, w * p2, c, size(x, 6)))
    return imgs
end
"""
    _fast_chunk(x::AbstractArray, ::Val{n}, ::Val{dim})

Type-stable and faster version of `MLUtils.chunk`.
"""
@inline _fast_chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function _fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, _fast_chunk(h, n))
end
@inline function _fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    return _fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end
@inline function _fast_chunk(
        x::GPUArraysCore.AnyGPUArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return copy(selectdim(x, dim, _fast_chunk(h, n)))
end

"""
    MultiHeadSelfAttention(
        embedding_dim::Int, 
        number_heads::Int; 
        qkv_bias::Bool=false,
        attention_dropout_rate::T=0.0f0, 
        projection_dropout_rate::T=0.0f0
    )

Multi-head self-attention layer

## Arguments

  - `embedding_dim`: number of input channels
  - `number_heads`: number of heads
  - `qkv_bias`: whether to use bias in the layer to get the query, key and value
  - `attention_dropout_rate`: dropout probability after the self-attention layer
  - `projection_dropout_rate`: dropout probability after the projection layer
"""
function MultiHeadSelfAttention(
    embedding_dim::Int, 
    number_heads::Int; 
    qkv_bias::Bool=false,
    attention_dropout_rate=0.0f0, 
    projection_dropout_rate=0.0f0
)

    qkv_layer = Lux.Dense(
        embedding_dim, 
        embedding_dim * number_heads * 3; 
        use_bias=qkv_bias
    )
    attention_dropout = Lux.Dropout(attention_dropout_rate)
    projection = Lux.Chain(
        Lux.Dense(embedding_dim * number_heads => embedding_dim), 
        Lux.Dropout(projection_dropout_rate)
    )

    layers = Lux.@compact(;
        number_heads, 
        qkv_layer, 
        attention_dropout,
        projection, 
        dispatch=:MultiHeadSelfAttention
    ) do x
        qkv = qkv_layer(x)

        q, k, v = _fast_chunk(qkv, Val(3), Val(1))
        y, _ = NNlib.dot_product_attention(
            q, k, v; fdrop=attention_dropout, nheads=number_heads)

        @return projection(y)
    end

    return layers
end

function SpatialAttention(
    embedding_dim::Int, 
    number_heads::Int; 
    qkv_bias::Bool=false,
    attention_dropout_rate=0.0f0, 
    projection_dropout_rate=0.0f0
)

    layers = Lux.@compact(
        attention = MultiHeadSelfAttention(
            embedding_dim, 
            number_heads; 
            qkv_bias=qkv_bias,
            attention_dropout_rate=attention_dropout_rate,
            projection_dropout_rate=projection_dropout_rate
        ),
    ) do x
        x = _flatten_spatial(x)
        x = attention(x)
        @return x
    end
