function val_step(
    ::Models.Stochastic,
    model::Models.FollmerStochasticInterpolant,
    base_samples,
    target_samples,
    field_conditioning,
    val_state,
    rng
)
    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Sample noise
    z_samples = Random.randn!(rng, similar(base_samples, size(base_samples)))
    _t_samples = Utils.reshape_scalar(t_samples, ndims(base_samples))
    z_samples = sqrt.(_t_samples) .* z_samples

    # Compute interpolated samples
    interpolated_samples, interpolated_samples_diff = get_interpolated_samples(
        base_samples,
        target_samples,
        z_samples,
        t_samples,
        model.interpolant_coefs
    )

    val_loss, st_, _ = compute_velocity_loss(
        model.velocity,
        val_state.ps,
        val_state.st,
        ((interpolated_samples, field_conditioning, t_samples), interpolated_samples_diff)
    )

    return val_loss, st_
end

"""
    _train_step(
        ::Models.Stochastic,
        model::Models.FollmerStochasticInterpolant,
        base_samples,
        target_samples,
        field_conditioning,
        train_state,
        rng
    )

    Train a stochastic interpolant generative model for one step.
"""
function _train_step(
    ::Models.Stochastic,
    model::Models.FollmerStochasticInterpolant,
    base_samples,
    target_samples,
    field_conditioning,
    train_state,
    rng
)
    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Sample noise
    z_samples = Random.randn!(rng, similar(base_samples, size(base_samples)))
    _t_samples = Utils.reshape_scalar(t_samples, ndims(base_samples))
    z_samples = sqrt.(_t_samples) .* z_samples

    # Compute interpolated samples
    interpolated_samples, interpolated_samples_diff = get_interpolated_samples(
        base_samples,
        target_samples,
        z_samples,
        t_samples,
        model.interpolant_coefs
    )

    # Compute gradients
    velocity_gs, velocity_loss, velocity_stats, velocity_train_state = get_gradients(
        ((interpolated_samples, field_conditioning, t_samples), interpolated_samples_diff),
        train_state,
        compute_velocity_loss
    )
    # Optimization
    train_state = Lux.Training.apply_gradients(velocity_train_state, velocity_gs)

    return velocity_loss, train_state, velocity_stats
end

"""
    train(
        ::Models.Stochastic,
        model::Models.FollmerStochasticInterpolant,
        data,
        config,
        rng = Random.default_rng();
        verbose = true,
        checkpoint = nothing
    )

    Train a Follmer stochastic interpolant generative model.
"""
function train(
    ::Models.Stochastic,
    model::Models.FollmerStochasticInterpolant,
    data,
    config,
    rng = Random.default_rng();
    verbose = true,
    checkpoint = nothing
)
    println("Training Follmer Stochastic Interpolant")

    # Get training parameters
    patience_counter = 0

    # Set model to train mode
    model.st = (; velocity = Lux.trainmode(model.st.velocity))

    # Get optimizers
    opt = get_optimizer(
        config.optimizer.type,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay
    )

    # Initialize train state
    train_state =
        Lux.Training.TrainState(model.velocity, model.ps.velocity, model.st.velocity, opt)

    iter = Utils.get_iter(config.training.num_epochs, verbose)

    train_data, val_data = split_data(data, 0.90, rng)
    val_dataloader = get_dataloader(
        val_data,
        config.training.batch_size,
        config.training.match_base_and_target
    )

    best_val_loss = Inf

    # Training loop
    for i in iter
        train_velocity_loss = 0.0

        # Prepare dataloader
        train_dataloader = get_dataloader(
            train_data,
            config.training.batch_size,
            config.training.match_base_and_target
        )

        # Loop over batches
        for batch in train_dataloader
            batch = batch .|> model.device

            # Training step
            train_velocity_loss, train_state, velocity_stats =
                _train_step(Models.Stochastic(), model, batch..., train_state, rng)
            train_velocity_loss += train_velocity_loss
        end

        train_velocity_loss = train_velocity_loss / length(train_dataloader)

        # Validation step
        val_state = (;
            ps = Lux.testmode(train_state.parameters),
            st = Lux.testmode(train_state.states)
        )
        val_velocity_loss = 0.0
        for batch in val_dataloader
            batch = batch .|> model.device

            _batch_val_velocity_loss, _ =
                val_step(Models.Stochastic(), model, batch..., val_state, rng)
            val_velocity_loss += _batch_val_velocity_loss
        end
        val_velocity_loss = val_velocity_loss / length(val_dataloader)

        if verbose && (i % 10 == 0)
            println("Epoch $i: Train loss = $train_velocity_loss, Val loss = $val_velocity_loss")
            println(" ")
        end

        if val_velocity_loss < best_val_loss
            best_val_loss = val_velocity_loss
            best_val_state = val_state
            patience_counter = 0

            # Save checkpoint
            if checkpoint !== nothing
                save_train_state(
                    best_val_state.ps |> CPU_DEVICE,
                    best_val_state.st |> CPU_DEVICE,
                    checkpoint
                )
            end
        else
            patience_counter += 1
            if patience_counter > config.training.patience
                break
            end
        end
    end

    model.ps = (; velocity = train_state.parameters)
    model.st = (; velocity = train_state.states)

    return model
end
