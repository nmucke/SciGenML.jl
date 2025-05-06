"""
    get_padding(padding::String, padding_size::Int)

    Get the padding function for the given padding type and size.
"""
function get_padding(padding::String, padding_size::Int)
    if padding == "constant"
        return a -> NNlib.pad_zeros(a, (padding_size, padding_size, 0, 0))

    elseif padding == "periodic"
        return a -> pad(a, :periodic, (padding_size, padding_size, 0, 0))

    elseif padding == "symmetric"
        return a -> pad(a, :symmetric, (padding_size, padding_size, 0, 0))

    elseif padding == "replicate"
        return a -> pad(a, :replicate, (padding_size, padding_size, 0, 0))

    elseif padding == "mirror"
        return a -> pad(a, :mirror, (padding_size, padding_size, 0, 0))

    else
        return a -> a
    end
end
