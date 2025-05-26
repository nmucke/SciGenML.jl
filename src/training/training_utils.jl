
SUPPORTED_OPTIMIZERS = ["adam", "adamw"]
SUPPORTED_LOSS_FUNCTIONS = ["mse"]

"""
    compute_velocity_loss(model, ps, st, (x, y))

    Compute the loss for a stochastic interpolant generative model.
"""
function compute_velocity_loss(model, ps, st, (x, y))
    y_pred, st_ = model(x, ps, st)
    loss = MSE_LOSS_FN(y_pred, y)
    return loss, st_, (; y_pred = y_pred)
end

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
    get_dataloader(
        data::NamedTuple,
        batch_size::Int,
        match_base_and_target::Bool=false
    )

    Get a dataloader from a named tuple.
"""
function get_dataloader(
    data::NamedTuple,
    batch_size::Int,
    match_base_and_target::Bool = false
)
    return get_dataloader(values(data)..., batch_size, match_base_and_target)
end

"""
    get_dataloader(
        base, 
        target, 
        batch_size::Int, 
        match_base_and_target::Bool=false
    )

    Get a dataloader from a base and target.
"""
function get_dataloader(
    base,
    target,
    batch_size::Int,
    match_base_and_target::Bool = false,
    rng = Random.default_rng()
)
    if !(base isa AbstractArray{<:Number})
        base = Random.rand!(rng, base, similar(target, size(target)))
    end

    if !match_base_and_target
        n_samples = size(target, ndims(target))
        base = selectdim(base, ndims(base), Random.randperm(rng, n_samples))
    end

    return DataLoaders.DataLoader(DataLoaders.shuffleobs((base, target)), batch_size)
end

"""
    get_dataloader(
        base,
        target,
        scalar_conditioning,
        batch_size::Int,
        match_base_and_target::Bool=false
    )

    Get a dataloader from a base, target, and conditioning.
"""
function get_dataloader(
    base,
    target,
    scalar_conditioning,
    batch_size::Int,
    match_base_and_target::Bool = false,
    rng = Random.default_rng()
)
    if !(base isa AbstractArray{<:Number})
        base = Random.rand!(rng, base, similar(target, size(target)))
    end

    if !match_base_and_target
        n_samples = size(target, ndims(target))
        base = selectdim(base, ndims(base), Random.randperm(rng, n_samples))
    end

    return DataLoaders.DataLoader(
        DataLoaders.shuffleobs((base, target, scalar_conditioning)),
        batch_size
    )
end

"""
    get_interpolated_samples(
        base_samples,
        target_samples,
        t_samples,
        interpolant_coefs
    )

    Get interpolated samples from a base, target, and time.
"""
function get_interpolated_samples(
    base_samples,
    target_samples,
    t_samples,
    interpolant_coefs
)

    # Compute interpolated samples
    interpolated_samples = Models.compute_interpolant(
        base_samples,
        target_samples,
        t_samples,
        interpolant_coefs
    )

    interpolated_samples_diff = Models.compute_interpolant_diff(
        base_samples,
        target_samples,
        t_samples,
        interpolant_coefs
    )

    return interpolated_samples, interpolated_samples_diff
end

"""
    get_interpolated_samples(
        base_samples,
        target_samples,
        z_samples,
        t_samples,
        interpolant_coefs
)

    Get interpolated samples from a base, target, and time.
"""
function get_interpolated_samples(
    base_samples,
    target_samples,
    z_samples,
    t_samples,
    interpolant_coefs
)
    # Compute interpolated samples
    interpolated_samples = Models.compute_interpolant(
        base_samples,
        target_samples,
        z_samples,
        t_samples,
        interpolant_coefs
    )

    interpolated_samples_diff = Models.compute_interpolant_diff(
        base_samples,
        target_samples,
        z_samples,
        t_samples,
        interpolant_coefs
    )

    return interpolated_samples, interpolated_samples_diff
end

"""
    get_gradients(
        batch,
        train_state,
        loss_fn,
    )

    Get gradients from a batch, train state, and loss function.
"""
function get_gradients(batch, train_state, loss_fn)
    # Compute gradients
    gs, loss, stats, train_state =
        Lux.Training.compute_gradients(Lux.AutoZygote(), loss_fn, batch, train_state)

    return gs, loss, stats, train_state
end
