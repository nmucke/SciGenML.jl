
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
    match_base_and_target::Bool = false,
    kwargs...
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
    split_data(data, split_ratio, rng = Random.default_rng())

    Split data into train and validation sets.
"""
function split_data(data, split_ratio, rng = Random.default_rng())
    n_samples = size(data.target, ndims(data.target))

    shuffled_indices = Random.randperm(rng, n_samples)

    train_indices = shuffled_indices[1:floor(Int, n_samples * split_ratio)]
    val_indices = shuffled_indices[(floor(Int, n_samples * split_ratio) + 1):end]

    train_data = Dict()
    val_data = Dict()

    for key in keys(data)
        samples = getproperty(data, key)

        if samples isa AbstractArray{<:Number}
            train_data[key] = Array(selectdim(samples, ndims(samples), train_indices))
            val_data[key] = Array(selectdim(samples, ndims(samples), val_indices))
        else
            train_data[key] = samples
            val_data[key] = samples
        end
    end

    train_data = NamedTuple(train_data)
    val_data = NamedTuple(val_data)

    return train_data, val_data
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

"""
    Checkpoint

A checkpoint for a model and other data.
"""
struct Checkpoint
    checkpoint_path::String
    config::Any

    # Constructor
    function Checkpoint(checkpoint_path::String, config::Any = nothing; create_new = true)

        # Check if path exists, if not create it
        if !isnothing(config) && create_new
            println("Creating new checkpoint...")

            if !isdir(checkpoint_path)
                println("Checkpoint path does not exist, creating it...")
                mkpath(checkpoint_path)
            end

            # Save config
            config_path = joinpath(checkpoint_path, "config.toml")
            Configurations.to_toml(config_path, config)

        else
            println("Loading checkpoint...")

            # Load config
            config_path = joinpath(checkpoint_path, "config.toml")
            config = Configurations.from_toml(Config.Hyperparameters, config_path)
        end

        return new(checkpoint_path, config)
    end
end

"""
    save_train_state(
        ps,
        st,
        checkpoint::Checkpoint
    )

    Save a train state to a checkpoint.
"""
function save_train_state(ps, st, checkpoint::Checkpoint)
    train_state_path = joinpath(checkpoint.checkpoint_path, "train_state.bson")
    BSON.bson(train_state_path, Dict("ps" => ps, "st" => st))
end

"""
    load_train_state(
        checkpoint::Checkpoint
    )

    Load a train state from a checkpoint.
"""
function load_train_state(checkpoint::Checkpoint)
    train_state_path = joinpath(checkpoint.checkpoint_path, "train_state.bson")
    return BSON.load(train_state_path)
end
