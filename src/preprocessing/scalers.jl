
"""
    reshape_scalar(scalar, new_dims)

Reshape a scalar to a new dimension.
"""
function reshape_scalar(scalar, new_dims)
    return reshape(scalar, new_dims)
end

"""
    get_dims_to_reduce(data)

Get the dimensions to reduce from the data. Outputs a tuple of dimensions to reduce.
"""
function get_dims_to_reduce(data, dim_to_not_reduce = nothing)
    if isnothing(dim_to_not_reduce)
        dim_to_not_reduce = ndims(data)-1
    end
    return Tuple(i for i in 1:ndims(data) if i != dim_to_not_reduce)
end

"""
    min_max_scaler_transform(
        data,
        min_val::Float32,
        max_val::Float32
    )

A scaler for data.
"""
function min_max_scaler_transform(data, min_val, max_val)
    return (data .- min_val) ./ (max_val .- min_val)
end

"""
    min_max_scaler_inverse_transform(
        data,
        min_val::Float32,
        max_val::Float32
    )

    Inverse transform for the min-max scaler.
"""
function min_max_scaler_inverse_transform(data, min_val, max_val)
    return data .* (max_val .- min_val) .+ min_val
end

"""
    MinMaxScaler

A min-max scaler for data.
"""
struct MinMaxScaler <: Preprocessor
    transform::Function
    inverse_transform::Function

    function MinMaxScaler(data)
        dims_to_reduce = get_dims_to_reduce(data)

        min_val = convert.(DEFAULT_TYPE, minimum(data, dims = dims_to_reduce))
        max_val = convert.(DEFAULT_TYPE, maximum(data, dims = dims_to_reduce))
        return new(
            data -> min_max_scaler_transform(data, min_val, max_val),
            data -> min_max_scaler_inverse_transform(data, min_val, max_val)
        )
    end

    function MinMaxScaler(min_val, max_val)
        min_val = convert.(DEFAULT_TYPE, min_val)
        max_val = convert.(DEFAULT_TYPE, max_val)
        return new(
            data -> min_max_scaler_transform(data, min_val, max_val),
            data -> min_max_scaler_inverse_transform(data, min_val, max_val)
        )
    end
end

"""
    standard_scaler_transform(
        data,
        mean,
        std
    )

A scaler for data.
"""
function standard_scaler_transform(data, mean, std)
    return (data .- mean) ./ std
end

"""
    standard_scaler_inverse_transform(
        data,
        mean,
        std
    )

    Inverse transform for the standard scaler.
"""
function standard_scaler_inverse_transform(data, mean, std)
    return data .* std .+ mean
end

"""
    StandardScaler

A scaler for data.
"""
struct StandardScaler <: Preprocessor
    transform::Function
    inverse_transform::Function

    function StandardScaler(data)
        dims_to_reduce = get_dims_to_reduce(data)

        mean = convert.(DEFAULT_TYPE, Statistics.mean(data, dims = dims_to_reduce))
        std = convert.(DEFAULT_TYPE, Statistics.std(data, dims = dims_to_reduce))
        return new(
            data -> standard_scaler_transform(data, mean, std),
            data -> standard_scaler_inverse_transform(data, mean, std)
        )
    end

    function StandardScaler(mean, std)
        mean = convert.(DEFAULT_TYPE, mean)
        std = convert.(DEFAULT_TYPE, std)
        return new(
            data -> standard_scaler_transform(data, mean, std),
            data -> standard_scaler_inverse_transform(data, mean, std)
        )
    end
end
