
"""
    _train_step(
        ::Models.Deterministic,
        model::Models.FlowMatching,
        base_samples,
        target_samples,
        train_state,
        rng
    )

    Train a stochastic interpolant generative model for one step.
"""
function _train_step(
    ::Models.Deterministic,
    model::Models.FlowMatching,
    base_samples,
    target_samples,
    train_state,
    rng
)

    ### Compute interpolated samples
    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Compute interpolated samples
    interpolated_samples = Models.compute_interpolant(
        base_samples,
        target_samples,
        t_samples,
        model.interpolant_coefs
    )

    interpolated_samples_diff = Models.compute_interpolant_diff(
        base_samples,
        target_samples,
        t_samples,
        model.interpolant_coefs
    )

    # Compute gradients
    velocity_gs, velocity_loss, velocity_stats, velocity_train_state =
        Lux.Training.compute_gradients(
            Lux.AutoZygote(),
            compute_velocity_loss,
            ((interpolated_samples, t_samples), interpolated_samples_diff),
            train_state.velocity
        )

    # Optimization
    train_state =
        (; velocity = Lux.Training.apply_gradients(velocity_train_state, velocity_gs),)

    return velocity_loss, train_state, velocity_stats
end

"""
    train(
        ::Models.Deterministic,
        model::Models.StochasticInterpolantGenerativeModel,
        data,
        config,
        rng = Random.default_rng();
        verbose = true
    )

    Train a stochastic interpolant generative model.
"""
function train(
    ::Models.Deterministic,
    model::Models.FlowMatching,
    data,
    config,
    rng = Random.default_rng();
    verbose = true
)
    println("Training Flow Matching Model")
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

        # Prepare batches
        x_batches, y_batches = prepare_batches(data, config.training.batch_size, rng)

        # Loop over batches
        for (x_batch, y_batch) in zip(x_batches, y_batches)
            x_batch = x_batch |> model.device
            y_batch = y_batch |> model.device

            # Training step
            velocity_loss, train_state, velocity_stats = _train_step(
                Models.Deterministic(),
                model,
                x_batch,
                y_batch,
                train_state,
                rng
            )
            velocity_loss += velocity_loss
        end

        velocity_loss = velocity_loss / length(x_batches)

        if verbose && (i % 50 == 0)
            println("Epoch $i: Velocity loss = $velocity_loss")
            println(" ")
        end
    end

    model.ps = (; velocity = train_state.velocity.parameters)
    model.st = (; velocity = train_state.velocity.states)

    return model
end
