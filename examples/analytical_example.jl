import Distributions
import Random
import Plots
import Zygote
import NNlib
import Statistics

num_samples = 10000;

mean = x -> abs.(x);
sigma = x -> 0.5f0 .* abs.(x);

x0_dist = Distributions.Normal(0.0f0, 1.0f0);
x1_dist = x -> Distributions.Normal(mean(x), sigma(x));

x0_samples = rand(x0_dist, num_samples);
x1_samples = map(x0_samples) do x0
    rand(x1_dist(x0), 1)
end;
x1_samples = vcat(x1_samples...)

Plots.histogram(x0_samples, label = "x0", normalize = :density)
Plots.histogram!(x1_samples, label = "x1", normalize = :density)

function obs_operator(x)
    return x
end

function target(x, obs, sigma_obs)
    likelihood = exp.(-0.5f0 * ((obs_operator(x) - obs) / sigma_obs)^2)
    prior = exp.(-0.5f0 * ((x - mean(x)) / sigma(x))^2)
    return likelihood .* prior
end

function metropolis_hastings(n_samples, obs, sigma_obs, n_burn)
    samples = zeros(Float32, n_samples)
    current = rand(x0_dist)  # Start from random initial point

    for i in 1:(n_burn + n_samples)
        # Propose new point
        proposal = current .+ 0.5f0 .* randn(Float32)

        # Calculate acceptance ratio
        # println(log_target(proposal, obs, sigma_obs))
        ratio = target(proposal, obs, sigma_obs) ./ target(current, obs, sigma_obs)

        # Accept or reject
        if rand(Float32) < ratio
            current = proposal
        end

        if i > n_burn
            samples[i - n_burn] = current
        end
    end

    return samples
end

alpha = t -> 1.0f0 .- t;
beta = t -> t .^ 2;
gamma = t -> 1.0f0 .- t;

alpha_diff = t -> -1.0f0;
beta_diff = t -> 2.0f0 .* t;
gamma_diff = t -> -1.0f0;

function drift_term(x, x0, t)
    mhat = alpha(t) .* x0 .+ beta(t) .* mean(x0)
    Chat = beta(t) .* sigma(x0) .^ 2 .+ t .* gamma(t) .^ 2

    out = alpha_diff(t) .* x0 .+ beta_diff(t) .* mean(x0)

    numerator =
        (beta(t) .* beta_diff(t) .* sigma(x0) .^ 2 .+ t .* gamma(t) .* gamma_diff(t))
    numerator = numerator .* (x .- mhat)

    out = out .+ numerator ./ (Chat .+ 1.0f-12)
    return out
end

function euler_maruyama(x0, t_steps, num_samples, diffusion_term, drift_term)
    x = zeros(Float32, length(t_steps), num_samples)
    x[1, :] = x0
    for i in 2:length(t_steps)
        dt = t_steps[i] - t_steps[i - 1]
        z = Random.randn(Float32, num_samples)
        diffusion_term_val = diffusion_term(t_steps[i - 1])
        x[i, :] = x[i - 1, :] .+ drift_term(x[i - 1, :], x0, t_steps[i - 1]) .* dt
        x[i, :] = x[i, :] .+ sqrt(dt) .* diffusion_term_val .* z
    end
    return x
end

# t_steps = sqrt.(0.0f0:0.1f0:1.0f0);
# pred_samples = euler_maruyama(x0_samples, t_steps, num_samples, gamma, drift_term);

# Plots.plot(t_steps[1:10:end], pred_samples[1:10:end, 1:100:end], label = "", alpha = 0.5)
# Plots.savefig("analytical_example_trajectories.png")

# Plots.histogram(x0_samples, label = "x0", normalize = :density)
# Plots.histogram!(x1_samples, label = "x1", normalize = :density, alpha = 0.75)
# Plots.histogram!(pred_samples[end, :], label = "SI", normalize = :density, alpha = 0.5)
# Plots.savefig("analytical_example.png")

sigma_obs = 1.5f0
num_mc_samples = 5

function compute_x1_pred(x, x0, t, drift_term, noise)
    prior_drift = drift_term(x, x0, t)
    x1_pred = x0 .+ prior_drift .* (1.0f0 .- t) .+ sqrt(1.0f0 ./ 3.0f0) .* noise
    # x1_pred = x0 .+ 0.5f0 .* (drift_term(x1_pred, x0, 1.0f0) + prior_drift) .* (1.0f0 .- t)
    # x1_pred = x1_pred .+ sqrt(1.0f0 ./ 3.0f0) .* noise
    return x1_pred
end;

function compute_log_likelihood(x, x0, y, t, drift_term, noise)
    x1 = compute_x1_pred(x, x0, t, drift_term, noise)
    return exp.(-0.5f0 .* ((x1 .- y) ./ sigma_obs) .^ 2) ./ (sigma_obs * sqrt(2.0f0 * pi))
end;

function compute_likelihood_weights(
    x,
    x0,
    y,
    t,
    drift_term,
    noise;
    num_mc_samples = num_mc_samples
)
    x0 = x0 .* ones(Float32, num_mc_samples)
    x = x .* ones(Float32, num_mc_samples)
    likelihood = compute_log_likelihood(x, x0, y, t, drift_term, noise)

    return NNlib.softmax(likelihood)
end;

function compute_log_likelihood_diff(
    x,
    x0,
    y,
    t,
    drift_term,
    noise;
    num_mc_samples = num_mc_samples
)
    x0 = x0 .* ones(Float32, num_mc_samples)
    x = x .* ones(Float32, num_mc_samples)

    log_likelihood_diff = Zygote.gradient(
        x -> sum(compute_log_likelihood(x, x0, y, t, drift_term, noise)),
        x
    )[1]
    return log_likelihood_diff
end;

function likelihood_score_fn(x, x0, y, t, drift_term; num_mc_samples = num_mc_samples)
    noise = randn(Float32, num_mc_samples)
    likelihood_weights = compute_likelihood_weights(
        x,
        x0,
        y,
        t,
        drift_term,
        noise;
        num_mc_samples = num_mc_samples
    )

    log_likelihood_diff = compute_log_likelihood_diff(
        x,
        x0,
        y,
        t,
        drift_term,
        noise;
        num_mc_samples = num_mc_samples
    )
    return sum(log_likelihood_diff .* likelihood_weights)
end;

function compute_expected_likelihood(
    x,
    x0,
    y,
    t,
    drift_term,
    noise;
    num_mc_samples = num_mc_samples
)
    out = [
        compute_log_likelihood(x, x0, y, t, drift_term, noise[j]) for j in 1:num_mc_samples
    ]
    return Statistics.mean(out)
end;

function log_like_score(x, x0, y, t, drift_term; num_mc_samples = num_mc_samples)
    expected_likelihood = zeros(Float32, length(x))
    for i in 1:length(x)
        noise = randn(Float32, num_mc_samples)
        expected_likelihood[i] =
            compute_expected_likelihood(x[i], x0[i], y, t, drift_term, noise)
    end

    expected_likelihood_diff = zeros(Float32, length(x))
    for i in 1:length(x)
        noise = randn(Float32, num_mc_samples)
        expected_likelihood_diff[i] = Zygote.gradient(
            x -> compute_expected_likelihood(x, x0[i], y, t, drift_term, noise),
            x[i]
        )[1]
    end

    return expected_likelihood_diff ./ expected_likelihood, expected_likelihood
end;

function posterior_drift_term(x, x0, y, t; flowdas = true)
    prior_drift = drift_term(x, x0, t)

    if beta(t) < 1.0f-10
        return prior_drift
    end

    if flowdas
        likelihood_score, expected_likelihood =
            log_like_score(x, x0, y, t, drift_term; num_mc_samples = 20)
        # likelihood_score = [likelihood_score_fn(x[i], x0[i], y, t, drift_term) for i in 1:length(x)]

        xi = t .* gamma(t) .* (gamma_diff(t) .* beta(t) .- beta_diff(t) .* gamma(t))
        likelihood_score = likelihood_score .* xi / beta(t)
    else
        sigma_obs_interpolant =
            sqrt.((alpha(t) .* sigma(x0)) .^ 2 .+ (beta(t) .* sigma_obs) .^ 2)

        likelihood_score = zeros(Float32, length(x))
        obs_interpolant = alpha(t) .* obs_operator(x0) .+ beta(t) .* y #.+  sigma_obs_interpolant .* randn(Float32, length(x))
        log_likelihood_score_fn =
            x ->
                - 0.5f0 .*
                ((obs_operator(x) .- obs_interpolant) ./ sigma_obs_interpolant) .^ 2 # .- log.(sigma_obs_interpolant .* sqrt(2.0f0 .* pi))
        likelihood_score += Zygote.gradient(x -> sum(log_likelihood_score_fn(x)), x)[1]

        # xi = t .* gamma(t) .* (gamma_diff(t) .* beta(t) .- beta_diff(t) .* gamma(t))
        # likelihood_score = likelihood_score .* xi ./ beta(t)

        # println(xi ./ beta(t))

    end

    return prior_drift .+ likelihood_score
end;

true_obs = 3.0f0;
obs = obs_operator(true_obs);

t_steps = 0.0f0:0.001f0:1.0f0;

x0_samples = rand(x0_dist, num_samples);

x = euler_maruyama(
    x[end, :],#x0_samples, 
    t_steps,
    num_samples,
    gamma,
    (x, x0, t) -> posterior_drift_term(x, x0, obs, t; flowdas = false)
);

# Generate MH samples
mh_samples = metropolis_hastings(num_samples, obs, sigma_obs, 100000);

# Plot MH samples
Plots.histogram(x0_samples, label = "x0", normalize = :density)
Plots.histogram!(x1_samples, label = "x1 prior", normalize = :density)
Plots.histogram!(x[end, :], label = "SI posterior", normalize = :density, alpha = 0.5)
Plots.histogram!(mh_samples, label = "MH posterior", normalize = :density, alpha = 0.5)
Plots.vline!([true_obs], label = "observation", color = "black", linewidth = 5)

Plots.plot(t_steps, x, label = "", alpha = 0.5)
Plots.savefig("analytical_example_posterior_trajectories.png")
