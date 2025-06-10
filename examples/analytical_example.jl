import Distributions
import Random
import Plots

num_samples = 10000

mean = x -> abs.(x)
sigma = x -> 0.5f0 .* abs.(x)

x0_dist = Distributions.Normal(0.0f0, 1.0f0)
x1_dist = x -> Distributions.Normal(mean(x), sigma(x))

x0_samples = rand(x0_dist, num_samples)
x1_samples = map(x0_samples) do x0
    rand(x1_dist(x0), 1)
end;
x1_samples = vcat(x1_samples...)

Plots.histogram(x0_samples, label = "x0", normalize = :density)
Plots.histogram!(x1_samples, label = "x1", normalize = :density)

alpha = t -> 1.0f0 .- t
beta = t -> t
gamma = t -> 1.0f0 .- t

alpha_diff = t -> -1.0f0
beta_diff = t -> 1.0f0
gamma_diff = t -> -1.0f0

drift_term = function (x, x0, t)
    mhat = alpha(t) .* x0 .+ beta(t) .* mean(x0)
    Chat = beta(t) .* sigma(x0) .^ 2 .+ t .* gamma(t) .^ 2

    out = alpha_diff(t) .* x0 .+ beta_diff(t) .* mean(x0)

    numerator =
        (beta(t) .* beta_diff(t) .* sigma(x0) .^ 2 .+ t .* gamma(t) .* gamma_diff(t))
    numerator = numerator .* (x .- mhat)

    denominator = Chat

    out = out .+ numerator ./ (denominator .+ 1e-12)

    return out
end

function euler_maruyama(x0, t_steps, num_samples, diffusion_term)
    x = zeros(Float32, length(t_steps), num_samples)
    dt = t_steps[2] - t_steps[1]
    x[1, :] = x0
    for i in 2:length(t_steps)
        z = Random.randn(Float32, num_samples)
        diffusion_term_val = diffusion_term(t_steps[i - 1])
        x[i, :] = x[i - 1, :] .+ drift_term(x[i - 1, :], x0, t_steps[i - 1]) .* dt
        x[i, :] = x[i, :] .+ sqrt(dt) .* diffusion_term_val .* z
    end
    return x
end

t_steps = 0.0f0:0.001f0:1.0f0;
pred_samples = euler_maruyama(x0_samples, t_steps, num_samples, gamma)

Plots.plot(t_steps[1:10:end], pred_samples[1:10:end, 1:100:end], label = "", alpha = 0.5)
Plots.savefig("analytical_example_trajectories.png")

Plots.histogram(x0_samples, label = "x0", normalize = :density)
Plots.histogram!(x1_samples, label = "x1", normalize = :density, alpha = 0.75)
Plots.histogram!(pred_samples[end, :], label = "SI", normalize = :density, alpha = 0.5)
Plots.savefig("analytical_example.png")

function likelihood_fn(x, x0, y, t)
    return y
end

function compute_likelihood_estimate(x, x0, y, t, prior_drift; num_mc_samples = 100)
    likelihood_samples = map(1:num_mc_samples) do i
        likelihood_fn(x, x0, y, t, prior_drift)
    end

    return mean(likelihood_samples)
end

function compute_ensemble_likelihood_estimate(
    x,
    x0,
    y,
    t,
    prior_drift;
    num_mc_samples = 100
)
    num_particles = size(x)[1]
    likelihood_samples = map(1:num_particles) do i
        compute_likelihood_estimate(
            x[i],
            x0,
            y,
            t,
            prior_drift;
            num_mc_samples = num_mc_samples
        )
    end

    return mean(likelihood_samples)
end

function posterior_drift_term(x, x0, y, t)
    prior_drift = drift_term(x, x0, t)

    lambda =
        1.0f0 ./
        (sqrt.(t) .* (gamma_diff(t) .* beta(t) .- gamma(t) .* beta_diff(t)) .+ 1e-12)

    likelihood = compute_likelihood_estimate(x, x0, y, t, prior_drift; num_mc_samples = 100)

    likelihood_term = likelihood ./ (lambda .* beta(t) .+ 1e-12)

    return prior_drift .+ likelihood_term
end
