import Distributions as Dist
import Random
import Plots
import Zygote
import NNlib
import Statistics
import KernelDensity as KD

##### Constants #####
num_samples = 10000;
num_mc_samples = 5
Id = [1.0f0 0.0f0; 0.0f0 1.0f0];
base_sample = [1.5f0, 1.5f0];

x_range = range(-5, 5, length = 150);
y_range = range(-5, 5, length = 150);

##### plotting #####
function get_kde_pdf(samples, num_points = 50)
    x = x_range #range(minimum(samples[1, :]), maximum(samples[1, :]), length=num_points);
    y = y_range #range(minimum(samples[2, :]), maximum(samples[2, :]), length=num_points);

    kde = KD.kde(transpose(samples));
    pdf = KD.pdf(kde, x, y);
    return pdf, x, y
end

function plot_pdf(samples, title = "", num_points = 50)
    pdf, x, y = get_kde_pdf(samples, num_points);

    Plots.heatmap(x, y, pdf, color = :viridis, title = title)

    Plots.plot!(x -> x, color = :red, linewidth = 2)
end

##### Distributions #####
# SI base distribution
x0_cov = [1.0f0 0.0f0; 0.0f0 1.0f0];
x0_dist = Dist.MvNormal([0.0f0, 0.0f0], x0_cov);
x0_samples = rand(x0_dist, num_samples);
x0_pdf, _, _ = get_kde_pdf(x0_samples);
x0_pdf_diagonal = [x0_pdf[i, i] for i in 1:size(x0_pdf, 1)]

# SI prior distribution
x1_mean = x -> [x[1] .^ 2, 1.0f0 * x[2]];

x1_cov = x -> [0.5f0 0.25f0; 0.25f0 0.5f0];
x1_cov_inv = x -> inv(x1_cov(x));

x1_dist = x0 -> Dist.MvNormal(x1_mean(x0), x1_cov(x0));
x1_samples = mapslices(x0 -> rand(x1_dist(base_sample)), x0_samples; dims = (1,));

x1_pdf, _, _ = get_kde_pdf(x1_samples);

# Get diagonal of x1_pdf
x1_pdf_diagonal = [x1_pdf[i, i] for i in 1:size(x1_pdf, 1)];

Plots.plot(
    plot_pdf(x0_samples, 100),
    plot_pdf(x1_samples, 100),
    layout = (1, 2),
    size = (1200, 400)
)

##### Observations #####
true_obs = [3.0f0, 3.0f0];
obs_operator = x -> x
obs = obs_operator(true_obs);

# obs_cov = [0.5f0 0.25f0; 0.25f0 0.5f0];
obs_cov = Id;

noise_dist = Dist.MvNormal(zeros(Float32, 2), obs_cov);
likelihood_function = (x, obs) -> Dist.pdf(noise_dist, obs - obs_operator(x))

likelihood_samples = rand(Dist.MvNormal(true_obs, obs_cov), num_samples);
likelihood_pdf, _, _ = get_kde_pdf(likelihood_samples);
likelihood_pdf_diagonal = [likelihood_pdf[i, i] for i in 1:size(likelihood_pdf, 1)];

##### Metropolis Hastings #####
function target(x, x0, obs)
    likelihood = likelihood_function(x, obs)
    prior = Dist.pdf(x1_dist(x0), x)
    return likelihood .* prior
end

function metropolis_hastings(n_samples, obs, x0, n_burn)
    samples = zeros(Float32, 2, n_samples)
    current = x0  # Start from random initial point

    for i in 1:(n_burn + n_samples)
        # Propose new point
        proposal = current .+ 0.5f0 .* randn(Float32, 2)

        # Calculate acceptance ratio
        # println(log_target(proposal, obs, sigma_obs))
        ratio = target(proposal, x0, obs) ./ target(current, x0, obs)

        # Accept or reject
        if rand(Float32) < ratio
            current = proposal
        end

        if i > n_burn
            samples[:, i - n_burn] = current
        end
    end

    return samples
end

# Generate MH samples
metropolis_hastings(num_samples, obs, base_sample, 10000)

mh_samples = map(1:10) do i
    metropolis_hastings(num_samples, obs, base_sample, 10000)
end
mh_samples = hcat(mh_samples...);

mh_pdf, _, _ = get_kde_pdf(mh_samples);

# Get diagonal of mh_pdf
mh_pdf_diagonal = [mh_pdf[i, i] for i in 1:size(mh_pdf, 1)];

Plots.plot(
    plot_pdf(mh_samples, "MH samples"),
    plot_pdf(x1_samples, "True samples"),
    plot_pdf(likelihood_samples, "Likelihood samples"),
    Plots.plot(
        [likelihood_pdf_diagonal, x1_pdf_diagonal, mh_pdf_diagonal],
        linewidth = 5,
        label = ["likelihood" "prior" "MH post"]
    ),
    layout = (2, 2),
    size = (1600, 1200),
    legend = :outertopright,
    title = ["MH" "Prior" "Likelihood samples" "slice"]
)

##### Stochastic Interpolant Prior #####
alpha = t -> 1.0f0 .- t;
beta = t -> t .^ 2;
gamma = t -> 1.0f0 .- t;

alpha_diff = t -> -1.0f0;
beta_diff = t -> 2.0f0 .* t;
gamma_diff = t -> -1.0f0;

function drift_term(x, x0, t)
    out = alpha_diff(t) .* x0 .+ beta_diff(t) .* x1_mean(x0)

    if t > 1.0f-8
        mhat = alpha(t) .* x0 .+ beta(t) .* x1_mean(x0)

        Chat = beta(t) .^ 2 .* x1_cov(x0) .+ t .* gamma(t) .^ 2 .* Id
        Chat_inv = inv(Chat)

        numerator = beta(t) .* beta_diff(t) .* x1_cov(x0)
        numerator = numerator .+ t .* gamma(t) .* gamma_diff(t) .* Id
        numerator = numerator * Chat_inv
        numerator = numerator * (x .- mhat)
    else
        numerator = zeros(Float32, 2)
    end

    out = out .+ numerator
    return out
end;

function eval_drift_term(x, x0, t)
    n_samples = size(x, 2)
    out = zeros(Float32, 2, n_samples)
    for i in 1:n_samples
        out[:, i] = drift_term(x[:, i], x0[:, i], t)
    end
    return out
end;

function euler_maruyama(x0, t_steps, num_samples, diffusion_term, drift_term)
    x = zeros(Float32, 2, length(t_steps), num_samples)

    if length(size(x0)) == 1
        x0 = repeat(x0, 1, num_samples)
    end

    x[:, 1, :] = x0
    for i in 2:length(t_steps)
        dt = t_steps[i] - t_steps[i - 1]
        z = Random.randn(Float32, 2, num_samples)
        diffusion_term_val = diffusion_term(t_steps[i - 1])
        x[:, i, :] = x[:, i - 1, :] .+ drift_term(x[:, i - 1, :], x0, t_steps[i - 1]) .* dt
        x[:, i, :] = x[:, i, :] .+ sqrt(dt) .* diffusion_term_val .* z
    end
    return x
end

t_steps = 0.0f0:0.01f0:1.0f0;
SI_prior_samples = euler_maruyama(base_sample, t_steps, num_samples, gamma, eval_drift_term);

SI_prior_pdf, _, _ = get_kde_pdf(SI_prior_samples[:, end, :]);
SI_prior_pdf_diagonal = [SI_prior_pdf[i, i] for i in 1:size(SI_prior_pdf, 1)];

Plots.plot(
    plot_pdf(SI_prior_samples[:, end, :], "SI prior"),
    plot_pdf(x1_samples, "True prior"),
    Plots.plot(
        [SI_prior_pdf_diagonal, x1_pdf_diagonal],
        linewidth = 5,
        label = ["SI prior" "True prior"]
    ),
    layout = (1, 3),
    size = (1600, 400)
)

##### Stochastic Interpolant Posterior#####

# function post_likelihood_function(x, obs, cov)
#     post_noise_dist = Dist.MvNormal(zeros(Float32, 2), cov);
#     return Dist.logpdf(post_noise_dist, obs - obs_operator(x))
# end;
function post_likelihood_function(x, obs, cov_inv)
    # jac_Hxt = Zygote.jacobian(x -> obs_operator(x), x)[1]

    # jac_Hxt = Zygote.jacobian(x -> obs_operator(x), x)[1]
    jac_Hxt = Id
    diff = obs - obs_operator(x)

    out = cov_inv * diff

    out = transpose(jac_Hxt) * out
    return out
end;

function posterior_drift_term(x, x0, y, t)
    prior_drift = eval_drift_term(x, x0, t)

    if beta(t) < 1.0f-8
        return prior_drift
    end

    likelihood_score = zeros(Float32, length(x))
    obs_interpolant = alpha(t) .* obs_operator(x0) .+ beta(t) .* y #.+ gamma(t) .* sqrt.(t) .* randn(Float32, size(x))

    # interpolant_cov = beta(t).^2 .* obs_cov# .+ Statistics.cov(obs_operator(x); dims=2)# .+ gamma(t).^2 .* Id

    # interpolant_cov = Statistics.cov(obs_operator(x), obs_operator(x); dims = 2)
    # interpolant_cov =
    #     interpolant_cov .- 2.0f0 .* beta(t) .* Statistics.cov(obs_operator(x), y; dims = 2)
    interpolant_cov = beta(t) .^ 2 .* obs_cov + gamma(t) .^ 2 .* t .* Id
    interpolant_cov_inv = inv(interpolant_cov)
    # interpolant_cov = interpolant_cov .+ gamma(t).^2 .* t .* Id

    # log_likelihood_score_fn =
    #     x -> post_likelihood_function(x, obs_interpolant, interpolant_cov)

    # likelihood_score = Zygote.gradient(x -> sum(log_likelihood_score_fn(x)), x)[1]

    # grad_Hxt = Zygote.gradient(x -> sum(obs_operator(x)), x)[1]

    # Evaluate likelihood score for each sample
    likelihood_score = zeros(Float32, size(x))
    for i in 1:size(x, 2)
        likelihood_score[:, i] =
            post_likelihood_function(x[:, i], obs_interpolant[:, i], interpolant_cov_inv)
    end

    # likelihood_score = post_likelihood_function(x, obs_interpolant, interpolant_cov_inv)

    # score_magnitude = sum(likelihood_score .^ 2, dims = 1)

    xi = t .* gamma(t) .* (beta_diff(t) .* gamma(t) .- beta(t) .* gamma_diff(t))
    likelihood_score = likelihood_score .* xi ./ beta(t)

    # theta = 1.0f0
    # diffusion_adaption = 1.0f0 .+ theta .* t .* (1.0f0 .- t) .* score_magnitude
    # diffusion_term = gamma(t) .^ 2 .* sqrt.(diffusion_adaption)

    return prior_drift .+ likelihood_score#, diffusion_term
end;

t_steps = 0.0f0:0.001f0:1.0f0;

obs_matrix = repeat(obs, 1, num_samples);

# function euler_maruyama_with_varying_diffusion(x0, t_steps, num_samples, drift_term)
#     x = zeros(Float32, 2, length(t_steps), num_samples)

#     if length(size(x0)) == 1
#         x0 = repeat(x0, 1, num_samples)
#     end

#     x[:, 1, :] = x0
#     for i in 2:length(t_steps)
#         dt = t_steps[i] - t_steps[i - 1]
#         z = Random.randn(Float32, 2, num_samples)
#         drift_term_val, diffusion_term_val = drift_term(x[:, i - 1, :], x0, t_steps[i - 1])
#         x[:, i, :] = x[:, i - 1, :] .+ drift_term_val .* dt
#         x[:, i, :] = x[:, i, :] .+ sqrt(dt) .* diffusion_term_val .* z
#     end
#     return x
# end

# x = euler_maruyama_with_varying_diffusion(
#     base_sample,
#     t_steps,
#     num_samples,
#     (x, x0, t) -> posterior_drift_term(x, x0, obs_matrix, t)
# );
x = euler_maruyama(
    base_sample,
    t_steps,
    num_samples,
    t -> 2.25f0 .* gamma(t),
    (x, x0, t) -> posterior_drift_term(x, x0, obs_matrix, t)
);

SI_posterior_pdf, _, _ = get_kde_pdf(x[:, end, :]);
SI_posterior_pdf_diagonal = [SI_posterior_pdf[i, i] for i in 1:size(SI_posterior_pdf, 1)];

exact_post_cov = inv(x1_cov_inv(base_sample) .+ Id)
exact_posteior_dist = Dist.MvNormal(
    exact_post_cov * (x1_cov_inv(base_sample)*obs_operator(x1_mean(base_sample)) .+ obs),
    exact_post_cov
)
exact_posterior_samples = rand(exact_posteior_dist, num_samples);
exact_posterior_pdf, _, _ = get_kde_pdf(exact_posterior_samples);
exact_posterior_pdf_diagonal =
    [exact_posterior_pdf[i, i] for i in 1:size(exact_posterior_pdf, 1)];

Plots.plot(
    plot_pdf(x[:, end, :], "SI posterior"),
    plot_pdf(x1_samples, "Prior"),
    plot_pdf(likelihood_samples, "Likelihood samples"),
    plot_pdf(mh_samples, "MH posterior"),
    plot_pdf(exact_posterior_samples, "Exact posterior"),
    Plots.plot(
        [
            SI_posterior_pdf_diagonal,
            x1_pdf_diagonal,
            likelihood_pdf_diagonal,
            mh_pdf_diagonal,
            exact_posterior_pdf_diagonal
        ],
        linewidth = 5,
        label = ["SI posterior" "Prior" "Likelihood samples" "MH posterior" "Exact posterior"]
    ),
    layout = (3, 2),
    size = (1600, 1600),
    legend = :outertopright,
    title = ["SI posterior" "Prior" "Likelihood samples" "MH posterior" "   Exact posterior"]
)
