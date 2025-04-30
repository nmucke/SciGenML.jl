
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
    n_samples = size(data.base, 2)
    shuffle_idx = Random.randperm(rng, n_samples)
    data = (base = data.base[:, shuffle_idx], target = data.target[:, shuffle_idx])

    x_batches = [
        data.base[:, i:(i + batch_size - 1)] for
        i in 1:batch_size:(n_samples - batch_size + 1)
    ]
    y_batches = [
        data.target[:, i:(i + batch_size - 1)] for
        i in 1:batch_size:(n_samples - batch_size + 1)
    ]

    return x_batches, y_batches
end

# """
#     prepare_batches(data::Dict{String, Distributions.Distribution}, batch_size::Int)

#     Prepare batches for training.
# """
# function prepare_batches(
#     data::Dict{String, Distributions.Distribution},
#     batch_size::Int,
#     rng::Random.AbstractRNG
# )
#     n_samples = size(data["base"], 2)

#     x_batches = [
#         rand(rng, data["base"], batch_size) for
#         i in 1:batch_size:(n_samples - batch_size + 1)
#     ]
#     y_batches = [
#         rand(rng, data["target"], batch_size) for
#         i in 1:batch_size:(n_samples - batch_size + 1)
#     ]

#     return x_batches, y_batches
# end
