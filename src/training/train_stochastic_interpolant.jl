
"""
    compute_score_loss(model, ps, st, (x, z, gamma))

    Compute the loss for a stochastic interpolant generative model.
"""
function compute_score_loss(model, ps, st, (x, z, gamma))
    y_pred, st_ = model(x, ps, st)

    gamma = Utils.reshape_scalar(gamma, ndims(x[1]))

    gamma = clamp.(gamma, 1.0f-5, Inf32)

    loss = 0.5f0 .* (y_pred .^ 2)
    loss = loss .+ 1.0f0 ./ gamma .* z .* y_pred
    loss = Statistics.mean(loss)

    return loss, st_, (; y_pred = y_pred)
end

"""
    _train_step(
        ::Models.Stochastic,
        model::Models.StochasticInterpolant,
        base_samples,
        target_samples,
        train_state,
        rng
    )

    Train a stochastic interpolant generative model for one step.
"""
function _train_step(
    ::Models.Stochastic,
    model::Models.StochasticInterpolant,
    base_samples,
    target_samples,
    train_state,
    rng
)
    ### Compute interpolated samples

    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Sample noise
    z_samples = Random.randn!(rng, similar(base_samples, size(base_samples)))

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
        ((interpolated_samples, t_samples), interpolated_samples_diff),
        train_state.velocity,
        compute_velocity_loss
    )
    score_gs, score_loss, score_stats, score_train_state = get_gradients(
        (
            (interpolated_samples, t_samples),
            z_samples,
            model.interpolant_coefs.gamma(t_samples)
        ),
        train_state.score,
        compute_score_loss
    )

    # Optimization
    train_state = (;
        velocity = Lux.Training.apply_gradients(velocity_train_state, velocity_gs),
        score = Lux.Training.apply_gradients(score_train_state, score_gs)
    )

    return velocity_loss, score_loss, train_state, velocity_stats, score_stats
end

"""
    _train_step(
        ::Models.Stochastic,
        model::Models.StochasticInterpolant,
        base_samples,
        target_samples,
        scalar_conditioning,
        train_state,
        rng
    )

    Train a stochastic interpolant generative model for one step.
"""
function _train_step(
    ::Models.Stochastic,
    model::Models.StochasticInterpolant,
    base_samples,
    target_samples,
    scalar_conditioning,
    train_state,
    rng
)
    ### Compute interpolated samples

    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Sample noise
    z_samples = Random.randn!(rng, similar(base_samples, size(base_samples)))

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
        ((interpolated_samples, scalar_conditioning, t_samples), interpolated_samples_diff),
        train_state.velocity,
        compute_velocity_loss
    )
    score_gs, score_loss, score_stats, score_train_state = get_gradients(
        (
            (interpolated_samples, scalar_conditioning, t_samples),
            z_samples,
            model.interpolant_coefs.gamma(t_samples)
        ),
        train_state.score,
        compute_score_loss
    )

    # Optimization
    train_state = (;
        velocity = Lux.Training.apply_gradients(velocity_train_state, velocity_gs),
        score = Lux.Training.apply_gradients(score_train_state, score_gs)
    )

    return velocity_loss, score_loss, train_state, velocity_stats, score_stats
end

"""
    _train_step(
        ::Models.Deterministic,
        model::Models.StochasticInterpolant,
        base_samples,
        target_samples,
        train_state,
        rng
    )

    Train a stochastic interpolant generative model for one step.
"""
function _train_step(
    ::Models.Deterministic,
    model::Models.StochasticInterpolant,
    base_samples,
    target_samples,
    train_state,
    rng
)
    ### Compute interpolated samples

    num_samples = size(base_samples)[end]

    # Generate random times
    t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

    # Sample noise
    z_samples = Random.randn!(rng, similar(base_samples, size(base_samples)))

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
        ((interpolated_samples, t_samples), interpolated_samples_diff),
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
        ::Models.Stochastic,
        model::Models.StochasticInterpolantGenerativeModel,
        data,
        config,
        rng = Random.default_rng();
        verbose = true
    )

    Train a stochastic interpolant generative model.
"""
function train(
    ::Models.Stochastic,
    model::Models.StochasticInterpolant,
    data,
    config,
    rng = Random.default_rng();
    verbose = true,
    checkpoint = nothing
)
    println("Training Stochastic Interpolant")

    # Set model to train mode
    model.st = (;
        velocity = Lux.trainmode(model.st.velocity),
        score = Lux.trainmode(model.st.score)
    )

    # Get optimizers
    opt = (;
        velocity = get_optimizer(
            config.optimizer.type,
            config.optimizer.learning_rate,
            config.optimizer.weight_decay
        ),
        score = get_optimizer(
            config.optimizer.type,
            config.optimizer.learning_rate,
            config.optimizer.weight_decay
        )
    )

    # Initialize train state
    train_state = (;
        velocity = Lux.Training.TrainState(
            model.velocity,
            model.ps.velocity,
            model.st.velocity,
            opt.velocity
        ),
        score = Lux.Training.TrainState(
            model.score,
            model.ps.score,
            model.st.score,
            opt.score
        )
    )

    iter = Utils.get_iter(config.training.num_epochs, verbose)

    # Training loop
    for i in iter
        velocity_loss = 0.0
        score_loss = 0.0

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
            velocity_loss, score_loss, train_state, velocity_stats, score_stats =
                _train_step(Models.Stochastic(), model, batch..., train_state, rng)
            velocity_loss += velocity_loss
            score_loss += score_loss
        end

        velocity_loss = velocity_loss / length(dataloader)
        score_loss = score_loss / length(dataloader)

        if verbose && (i % 10 == 0)
            println("Epoch $i: Velocity loss = $velocity_loss, Score loss = $score_loss")
            println(" ")
        end

        # Save checkpoint
        if checkpoint !== nothing
            Checkpoint.save_train_state(train_state, checkpoint)
        end
    end

    model.ps =
        (; velocity = train_state.velocity.parameters, score = train_state.score.parameters)
    model.st = (; velocity = train_state.velocity.states, score = train_state.score.states)

    return model
end

"""
    train(
        ::Models.Deterministic,
        model::Models.StochasticInterpolantGenerativeModel,
        data,
        config,
        rng = Random.default_rng();
        verbose = true,
        checkpoint = nothing
    )

    Train a stochastic interpolant generative model.
"""
function train(
    ::Models.Deterministic,
    model::Models.StochasticInterpolant,
    data,
    config,
    rng = Random.default_rng();
    verbose = true,
    checkpoint = nothing
)
    println("Training Stochastic Interpolant")

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
