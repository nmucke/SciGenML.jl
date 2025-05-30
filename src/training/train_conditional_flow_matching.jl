
"""
    _train_step(
        ::Models.Deterministic,
        model::Models.ConditionalFlowMatching,
        base_samples,
        target_samples,
        scalar_conditioning_samples,
        train_state,
        rng
    )

    Train a stochastic interpolant generative model for one step.
"""
function _train_step(
    ::Models.Deterministic,
    model::Models.ConditionalFlowMatching,
    base_samples,
    target_samples,
    scalar_conditioning_samples,
    train_state,
    rng
)

    ### Compute interpolated samples
    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Compute interpolated samples
    interpolated_samples, interpolated_samples_diff = get_interpolated_samples(
        base_samples,
        target_samples,
        t_samples,
        model.interpolant_coefs
    )

    # Replace samples with unconditional value
    replacement_mask =
        Random.rand(rng, size(scalar_conditioning_samples)...) .<
        model.replacement_probability
    scalar_conditioning_samples[replacement_mask] .= model.unconditional_condition

    # Compute gradients
    velocity_gs, velocity_loss, velocity_stats, velocity_train_state = get_gradients(
        (
            (interpolated_samples, scalar_conditioning_samples, t_samples),
            interpolated_samples_diff
        ),
        train_state.velocity,
        compute_velocity_loss
    )

    # Optimization
    train_state =
        (; velocity = Lux.Training.apply_gradients(velocity_train_state, velocity_gs),)

    return velocity_loss, train_state, velocity_stats
end

"""
    train(
        ::Models.Deterministic,
        model::Models.ConditionalFlowMatching,
        data,
        config,
        rng = Random.default_rng();
        verbose = true
    )

    Train a stochastic interpolant generative model.
"""
function train(
    ::Models.Deterministic,
    model::Models.ConditionalFlowMatching,
    data,
    config,
    rng = Random.default_rng();
    checkpoint = nothing,
    verbose = true
)
    println("Training Conditional Flow Matching Model")
    # Set model to train mode
    model.st = (; velocity = Lux.trainmode(model.st.velocity),)

    # Get optimizers
    opt = (;
        velocity = get_optimizer(
            config.optimizer.type,
            config.optimizer.learning_rate,
            config.optimizer.weight_decay
        ),)

    # Initialize train state
    train_state = (;
        velocity = Lux.Training.TrainState(
            model.velocity,
            model.ps.velocity,
            model.st.velocity,
            opt.velocity
        ),)

    iter = Utils.get_iter(config.training.num_epochs, verbose)

    # Training loop
    for i in iter
        velocity_loss = 0.0

        # Prepare dataloader
        dataloader = get_dataloader(
            data,
            config.training.batch_size,
            config.training.match_base_and_target
        )

        # Loop over batches
        for batch in dataloader
            batch = batch .|> model.device

            # Training step
            velocity_loss, train_state, velocity_stats =
                _train_step(Models.Deterministic(), model, batch..., train_state, rng)
            velocity_loss += velocity_loss
        end

        velocity_loss = velocity_loss / length(dataloader)

        if verbose && (i % 10 == 0)
            println("Epoch $i: Velocity loss = $velocity_loss")
            println(" ")
        end

        # Save checkpoint
        if checkpoint !== nothing
            Checkpoint.save_train_state(train_state, checkpoint)
        end
    end

    model.ps = (; velocity = train_state.velocity.parameters)
    model.st = (; velocity = train_state.velocity.states)

    return model
end
