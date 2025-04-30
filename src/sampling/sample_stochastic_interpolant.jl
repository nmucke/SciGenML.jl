
"""
    sample(
        model::Models.StochasticInterpolantGenerativeModel,
        num_samples::Int,
        num_steps::Int,
        rng::Random.AbstractRNG = Random.default_rng()
    )

    Sample from a stochastic interpolant generative model using the forward Euler method.
"""
function sample(
    model::Models.StochasticInterpolantGenerativeModel,
    num_samples::Int,
    num_steps::Int;
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true
)
    x_samples = rand(rng, Distributions.Normal(0.0, 1.0), (1, num_samples)) .|> DEFAULT_TYPE

    dt = 1.0 / num_steps |> DEFAULT_TYPE

    t_i = zeros(DEFAULT_TYPE, (1, num_samples))

    if verbose
        iter = ProgressBars.ProgressBar(1:num_steps)
    else
        iter = 1:num_steps
    end

    for i in iter
        drift, st_ = model.drift_model((x_samples, t_i), model.ps, model.st)
        model.st = st_

        x_samples = x_samples .+ drift .* dt
        t_i = t_i .+ dt
    end

    return x_samples
end
