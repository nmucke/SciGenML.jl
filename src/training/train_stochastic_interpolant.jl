
const MSE_LOSS_FN = Lux.MSELoss()
const ZERO_TOL = 1.0f-12

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
    compute_score_loss(model, ps, st, (x, z, gamma))

    Compute the loss for a stochastic interpolant generative model.
"""
function compute_score_loss(model, ps, st, (x, z, gamma))
    y_pred, st_ = model(x, ps, st)

    loss = y_pred .^ 2
    loss = loss .+ 1.0f0 ./ (gamma .+ ZERO_TOL) .* z .^ 2
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
    z_samples = Random.randn!(rng, similar(base_samples, (1, num_samples)))

    # Compute interpolated samples
    interpolated_samples = Models.compute_interpolant(
        base_samples,
        target_samples,
        z_samples,
        t_samples,
        model.interpolant_coefs
    )

    interpolated_samples_diff = Models.compute_interpolant_diff(
        base_samples,
        target_samples,
        z_samples,
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

    score_gs, score_loss, score_stats, score_train_state = Lux.Training.compute_gradients(
        Lux.AutoZygote(),
        compute_score_loss,
        (
            (interpolated_samples, t_samples),
            z_samples,
            model.interpolant_coefs.gamma(t_samples)
        ),
        train_state.score
    )

    # Optimization
    train_state = (;
        velocity = Lux.Training.apply_gradients(velocity_train_state, velocity_gs),
        score = Lux.Training.apply_gradients(score_train_state, score_gs)
    )

    return velocity_loss, score_loss, train_state, velocity_stats, score_stats
end

"""
    train(
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
    verbose = true
)

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

    if verbose
        println("Training model for $(config.training.num_epochs) epochs...")
        iter = ProgressBars.ProgressBar(1:config.training.num_epochs)
    else
        iter = 1:config.training.num_epochs
    end

    # Training loop
    for i in iter
        velocity_loss = 0.0
        score_loss = 0.0

        # Prepare batches
        x_batches, y_batches = prepare_batches(data, config.training.batch_size, rng)

        # Loop over batches
        for (x_batch, y_batch) in zip(x_batches, y_batches)
            # Training step
            velocity_loss, score_loss, train_state, velocity_stats, score_stats =
                _train_step(Models.Stochastic(), model, x_batch, y_batch, train_state, rng)
            velocity_loss += velocity_loss
            score_loss += score_loss
        end

        velocity_loss = velocity_loss / length(x_batches)
        score_loss = score_loss / length(x_batches)

        if verbose && (i % 50 == 0)
            println("Epoch $i: Velocity loss = $velocity_loss, Score loss = $score_loss")
            println(" ")
        end
    end

    model.ps =
        (; velocity = train_state.velocity.parameters, score = train_state.score.parameters)
    model.st = (; velocity = train_state.velocity.states, score = train_state.score.states)

    return model
end
