"""
    get_padding(padding::String, padding_size::Int)

    Get the padding function for the given padding type and size.
"""
function get_padding(padding::String, padding_size::Int)
    if padding == "constant"
        return a -> NNlib.pad_zeros(a, (padding_size, padding_size, 0, 0))

    elseif padding == "periodic"
        return a -> NNlib.pad_circular(a, padding_size; dims = (1, 2))
    else
        throw(ArgumentError("Invalid padding type: $padding. Must be one of: constant, periodic"))
    end
end
