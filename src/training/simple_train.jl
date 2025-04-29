
"""
    simple_train(;
        model,
        ps,
        st,
        data,
        opt = DEFAULT_OPTIMIZER,
        loss_fn = DEFAULT_LOSS_FN,
        num_epochs = DEFAULT_NUM_EPOCHS,
        batch_size = DEFAULT_BATCH_SIZE,
        verbose = true
    )

    model: The model to train.
    ps: The parameters of the model.
    st: The state of the model.
    opt: The optimizer to use.
    loss_fn: The loss function to use.
    data: The data to train on.
    num_epochs: The number of epochs to train for.
    batch_size: The batch size to use.
    verbose: Whether to print the loss during training.

    Returns the parameters and states of the model after training.

Train a model for a given number of epochs using a simple training loop.
"""

function simple_train(;
        model,
        ps,
        st,
        data,
        opt = DEFAULT_OPTIMIZER,
        loss_fn = DEFAULT_LOSS_FN,
        num_epochs = DEFAULT_NUM_EPOCHS,
        batch_size = DEFAULT_BATCH_SIZE,
        verbose = true
)

    # Compute loss
    function compute_loss(model, ps, st, (x, y))
        y_pred, st_ = model(x, ps, st)
        loss = loss_fn(y_pred, y)
        return loss, st_, (; y_pred = y_pred)
    end

    # Initialize train state
    train_state = Lux.Training.TrainState(model, ps, st, opt)

    if verbose
        println("Training model for $num_epochs epochs...")
        iter = ProgressBars.ProgressBar(1:num_epochs)
    else
        iter = 1:num_epochs
    end

    loss = 0.0
    for i in iter

        # Shuffle data indices
        n_samples = size(data.x, 2)
        shuffle_idx = Random.randperm(n_samples)
        data = (x = data.x[:, shuffle_idx], y = data.y[:, shuffle_idx])

        # Create batches
        x_batches = [data.x[:, i:(i + batch_size - 1)]
                     for
                     i in 1:batch_size:(n_samples - batch_size + 1)]
        y_batches = [data.y[:, i:(i + batch_size - 1)]
                     for
                     i in 1:batch_size:(n_samples - batch_size + 1)]

        for (x_batch, y_batch) in zip(x_batches, y_batches)
            # Compute gradients
            gs, loss,
            stats,
            train_state = Lux.Training.compute_gradients(
                Lux.AutoZygote(), compute_loss, (x_batch, y_batch), train_state
            )

            # Optimization
            train_state = Lux.Training.apply_gradients(train_state, gs)
        end

        if verbose && i % 10 == 0
            println("Epoch $i: loss = $loss")
        end
    end

    return train_state.parameters, train_state.states
end
