
SUPPORTED_OPTIMIZERS = ["adam", "adamw"]
SUPPORTED_LOSS_FUNCTIONS = ["mse"]

"""
    get_optimizer(type::String, lr::Float32, lambda::Float32)

    Get an optimizer from a string.
"""
function get_optimizer(type::String, lr, weight_decay)
    if type == "adam"
        return Optimisers.Adam(; eta = lr)
    elseif type == "adamw"
        return Optimisers.AdamW(; eta = lr, lambda = weight_decay)
    else
        throw(ArgumentError("Invalid optimizer type: $type. Supported optimizers: $SUPPORTED_OPTIMIZERS"))
    end
end

"""
    prepare_batches(data::Dict{String, AbstractArray}, batch_size::Int)

    Prepare batches for training.
"""
function prepare_batches(data, batch_size::Int, rng::Random.AbstractRNG)
    n_samples = size(data.target, ndims(data.target))

    if data.base isa AbstractArray{<:Number}
        data = (
            base = selectdim(data.base, ndims(data.base), Random.randperm(rng, n_samples)),
            target = selectdim(
                data.target,
                ndims(data.target),
                Random.randperm(rng, n_samples)
            )
        )
    else
        data = (
            base = Random.rand!(rng, data.base, similar(data.target, size(data.target))),
            target = selectdim(
                data.target,
                ndims(data.target),
                Random.randperm(rng, n_samples)
            )
        )
    end

    x_batches = [
        selectdim(data.base, ndims(data.base), i:min(i + batch_size - 1, n_samples)) for
        i in 1:batch_size:n_samples
    ]
    y_batches = [
        selectdim(data.target, ndims(data.target), i:min(i + batch_size - 1, n_samples)) for i in 1:batch_size:n_samples
    ]

    return x_batches, y_batches
end
