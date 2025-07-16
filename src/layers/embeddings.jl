"""
    sinusoidal_embedding(
        x::AbstractArray{AbstractFloat, 4},
        min_freq::AbstractFloat,
        max_freq::AbstractFloat,
        embedding_dims::Int
    )

Embed the noise variances to a sinusoidal embedding with the given frequency
range and embedding dimensions.

Based on https://yng87.page/en/blog/2022/lux-ddim/.
"""
function sinusoidal_embedding(
    x,
    embedding_dims::Int,
    min_freq::AbstractFloat = 1.0f0,
    max_freq::AbstractFloat = 1000.0f0
)

    # get device of x
    dev = Lux.get_device(x)

    # define frequencies
    # LinRange requires @adjoint when used with Zygote
    # Instead we manually implement range.
    lower = log(min_freq)
    upper = log(max_freq)
    n = div(embedding_dims, 2)
    d = (upper - lower) / (n - 1)
    freqs = exp.(lower:d:upper) |> dev

    angular_speeds = reshape(2.0f0 * DEFAULT_TYPE(pi) .* freqs, (length(freqs), 1))

    return cat(sin.(angular_speeds .* x), cos.(angular_speeds .* x); dims = 1)
end
