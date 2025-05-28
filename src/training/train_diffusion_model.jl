
"""
    _train_step(
        ::Models.Deterministic,
        model::Models.ScoreBasedDiffusionModel,
        base_samples,
        target_samples,
        conditioning,
        train_state,
        rng
    )

    Train a score-based diffusion generative model for one step via flow matching.
"""
function _train_step(
    model::Models.ScoreBasedDiffusionModel,
    base_samples,
    target_samples,
    conditioning,
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

    # Compute gradients
    velocity_gs, velocity_loss, velocity_stats, velocity_train_state = get_gradients(
        ((interpolated_samples, conditioning, t_samples), interpolated_samples_diff),
        train_state.velocity,
        compute_velocity_loss
    )

    # Optimization
    train_state =
        (; velocity = Lux.Training.apply_gradients(velocity_train_state, velocity_gs),)

    return velocity_loss, train_state, velocity_stats
end

"""
    _train_step(
        ::Models.Deterministic,
        model::Models.ScoreBasedDiffusionModel,
        base_samples,
        target_samples,
        train_state,
        rng
    )

    Train a score-based diffusion generative model for one step via flow matching.
"""
function _train_step(
    model::Models.ScoreBasedDiffusionModel,
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
    interpolated_samples, interpolated_samples_diff = get_interpolated_samples(
        base_samples,
        target_samples,
        t_samples,
        model.interpolant_coefs
    )

    # Compute gradients
    velocity_gs, velocity_loss, velocity_stats, velocity_train_state = get_gradients(
        model,
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
        model::Models.ScoreBasedDiffusionModel,
        data,
        config,
        rng = Random.default_rng();
        verbose = true
    )

    Train a score-based diffusion generative model vis flow matching.
"""
function train(
    ::Models.Stochastic,
    model::Models.ScoreBasedDiffusionModel,
    data,
    config,
    rng = Random.default_rng();
    verbose = true
)
    println("Training Denoising Diffusion Model")
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
                _train_step(model, batch..., train_state, rng)
            velocity_loss += velocity_loss
        end

        velocity_loss = velocity_loss / length(dataloader)

        if verbose && (i % 10 == 0)
            println("Epoch $i: Velocity loss = $velocity_loss")
            println(" ")
        end
    end

    model.ps = (; velocity = train_state.velocity.parameters)
    model.st = (; velocity = train_state.velocity.states)

    return model
end

# """
#     compute_score_loss(model, ps, st, (x, z, gamma))

#     Compute the loss for a stochastic interpolant generative model.
# """
# function compute_diffusion_score_loss(model, ps, st, (x, z, alpha))
#     y_pred, st_ = model(x, ps, st)

#     alpha = Utils.reshape_scalar(alpha, ndims(x[1]))

#     alpha = clamp.(alpha, 1.0f-5, Inf32)

#     loss = (y_pred .+ z ./ alpha) .^ 2
#     loss = Statistics.mean(loss)

#     return loss, st_, (; y_pred = y_pred)
# end

# """
#     _train_step(
#         model::Models.ScoreBasedDiffusionModel,
#         base_samples,
#         target_samples,
#         train_state,
#         rng
#     )

#     Train a score-based diffusion model for one step.
# """
# function _train_step(
#     model::Models.ScoreBasedDiffusionModel,
#     base_samples,
#     target_samples,
#     train_state,
#     rng
# )
#     ### Compute interpolated samples
#     num_samples = size(base_samples)[end]

#     # Generate random times
#     t_samples = Random.rand!(rng, similar(base_samples, (1, num_samples)))

#     # Compute interpolated samples
#     interpolated_samples = Models.compute_interpolant(
#         base_samples,
#         target_samples,
#         t_samples,
#         model.interpolant_coefs
#     )

#     interpolated_samples_diff = Models.compute_interpolant_diff(
#         base_samples,
#         target_samples,
#         t_samples,
#         model.interpolant_coefs
#     )

#     # Compute gradients
#     velocity_gs, velocity_loss, velocity_stats, velocity_train_state =
#         Lux.Training.compute_gradients(
#             Lux.AutoZygote(),
#             compute_velocity_loss,
#             ((interpolated_samples, t_samples), interpolated_samples_diff),
#             train_state
#         )

#     # Optimization
#     train_state = Lux.Training.apply_gradients(velocity_train_state, velocity_gs)

#     return velocity_loss, train_state , velocity_stats
# end

# """
#     train(
#         ::Models.Stochastic,
#         model::Models.ScoreBasedDiffusionModel,
#         data,
#         config,
#         rng = Random.default_rng();
#         verbose = true
#     )

#     Train a stochastic interpolant generative model.
# """
# function train(
#     ::Models.Stochastic,
#     model::Models.ScoreBasedDiffusionModel,
#     data,
#     config,
#     rng = Random.default_rng();
#     verbose = true
# )

#     # Set model to train mode
#     model.st = (;
#         velocity = Lux.trainmode(model.st.velocity)
#     )

#     # Get optimizers
#     opt = get_optimizer(
#         config.optimizer.type,
#         config.optimizer.learning_rate,
#         config.optimizer.weight_decay
#     )

#     # Initialize train state
#     train_state = Lux.Training.TrainState(
#         model.velocity,
#         model.ps.velocity,
#         model.st.velocity,
#         opt
#     )

#     iter = Utils.get_iter(config.training.num_epochs, verbose)

#     # Training loop
#     for i in iter
#         velocity_loss = 0.0

#         # Prepare batches
#         x_batches, y_batches = prepare_batches(data, config.training.batch_size, rng)

#         # Loop over batches
#         for (x_batch, y_batch) in zip(x_batches, y_batches)
#             x_batch = x_batch |> model.device
#             y_batch = y_batch |> model.device

#             # Training step
#             velocity_loss, train_state, velocity_stats =
#                 _train_step(model, x_batch, y_batch, train_state, rng)
#             velocity_loss += velocity_loss
#         end

#         velocity_loss = velocity_loss / length(x_batches)

#         if verbose && (i % 10 == 0)
#             println("Epoch $i: Velocity loss = $velocity_loss")
#             println(" ")
#         end
#     end

#     model.st = (; velocity = train_state.states)

#     return model
# end
