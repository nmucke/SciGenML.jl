
const LOSS_FN = Lux.MSELoss()

"""
    compute_loss(model, ps, st, (x, y))

    Compute the loss for a stochastic interpolant generative model.
"""
function compute_loss(model, ps, st, (x, y))
    y_pred, st_ = model(x, ps, st)
    loss = LOSS_FN(y_pred, y)
    return loss, st_, (; y_pred = y_pred)
end

"""
    stochastic_interpolant_train_step(
        model::Models.StochasticInterpolantGenerativeModel,
        base_samples::AbstractArray,
        target_samples::AbstractArray,
        train_state::Lux.Training.TrainState,
        rng::Random.AbstractRNG
    )

    Train a stochastic interpolant generative model for one step.
"""
function stochastic_interpolant_train_step(
    model::Models.StochasticInterpolantGenerativeModel,
    base_samples::AbstractArray,
    target_samples::AbstractArray,
    train_state::Lux.Training.TrainState,
    rng::Random.AbstractRNG
)
    # Compute interpolated samples
    t_samples =
        rand(rng, Distributions.Uniform(0.0, 1.0), size(base_samples)) .|> DEFAULT_TYPE

    interpolated_samples = Models.compute_interpolant(
        base_samples,
        target_samples,
        model.interpolant_coefs,
        t_samples
    )

    interpolated_samples_diff = Models.compute_interpolant_diff(
        base_samples,
        target_samples,
        model.interpolant_coefs,
        t_samples
    )

    # Compute gradients
    gs, loss, stats, train_state = Lux.Training.compute_gradients(
        Lux.AutoZygote(),
        compute_loss,
        ((interpolated_samples, t_samples), interpolated_samples_diff),
        train_state
    )

    # Optimization
    train_state = Lux.Training.apply_gradients(train_state, gs)

    return loss, train_state, stats
end

function train(
    model::Models.StochasticInterpolantGenerativeModel,
    data,
    config,
    rng = Random.default_rng();
    verbose = true
)
    model.st = Lux.trainmode(model.st)

    opt = get_optimizer(
        config.optimizer.type,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay
    )

    # Initialize train state
    train_state = Lux.Training.TrainState(model.drift_model, model.ps, model.st, opt)

    if verbose
        println("Training model for $(config.training.num_epochs) epochs...")
        iter = ProgressBars.ProgressBar(1:config.training.num_epochs)
    else
        iter = 1:config.training.num_epochs
    end

    for i in iter
        loss = 0.0

        x_batches, y_batches = prepare_batches(data, config.training.batch_size, rng)
        for (x_batch, y_batch) in zip(x_batches, y_batches)
            batch_loss, train_state, stats =
                stochastic_interpolant_train_step(model, x_batch, y_batch, train_state, rng)
            loss += batch_loss
        end

        loss = loss / length(x_batches)

        if verbose && (i % 5 == 0)
            println("Epoch $i: loss = $loss")
            println("")
        end
    end

    model.ps = train_state.parameters
    model.st = train_state.states

    return model
end
