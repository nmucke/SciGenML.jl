import Distributions as Dist
import Statistics as Stats
import KernelDensity as KD
import Plots
import Zygote
import Random

include("2d_analytical/config.jl");
include("2d_analytical/utils.jl");
include("2d_analytical/metropolis_hastings.jl");
include("2d_analytical/stochastic_interpolant.jl");

##### Generate Samples #####
x0_samples = rand(x0_dist, num_samples);
x1_samples = sample_conditional(base_sample, x1_dist, num_samples);
likelihood_samples = sample_conditional(true_x, likelihood_dist, num_samples);
posterior_samples = rand(true_posterior_dist, num_samples);

x0_pdf = get_kde_pdf(x0_samples);
x1_pdf = get_kde_pdf(x1_samples);
likelihood_pdf = get_kde_pdf(likelihood_samples);
posterior_pdf = get_kde_pdf(posterior_samples);

x0_pdf_diagonal = get_pdf_diagonal(x0_pdf);
x1_pdf_diagonal = get_pdf_diagonal(x1_pdf);
likelihood_pdf_diagonal = get_pdf_diagonal(likelihood_pdf);
posterior_pdf_diagonal = get_pdf_diagonal(posterior_pdf);

Plots.plot(
    plot_pdf(x0_pdf, "x0 pdf"),
    plot_pdf(x1_pdf, "x1 pdf"),
    plot_pdf(likelihood_pdf, "likelihood pdf"),
    plot_pdf(posterior_pdf, "posterior pdf"),
    Plots.plot(
        [x0_pdf_diagonal, x1_pdf_diagonal, likelihood_pdf_diagonal, posterior_pdf_diagonal],
        linewidth = 5,
        label = ["x0 pdf" "x1 pdf" "likelihood pdf" "posterior pdf"]
    ),
    layout = (3, 2),
    size = (1200, 800)
)

##### Metropolis Hastings #####
mh_samples = metropolis_hastings(num_samples, obs, base_sample, 10000);

mh_pdf = get_kde_pdf(mh_samples);
mh_pdf_diagonal = get_pdf_diagonal(mh_pdf);

Plots.plot(
    plot_pdf(mh_pdf, "MH pdf"),
    plot_pdf(x1_pdf, "x1 pdf"),
    plot_pdf(likelihood_pdf, "likelihood pdf"),
    plot_pdf(posterior_pdf, "posterior pdf"),
    Plots.plot(
        [x1_pdf_diagonal, likelihood_pdf_diagonal, mh_pdf_diagonal, posterior_pdf_diagonal],
        linewidth = 5,
        label = ["x1 pdf" "likelihood pdf" "MH pdf" "posterior pdf"]
    ),
    layout = (3, 2),
    size = (1200, 800)
)

##### Stochastic Interpolant Prior #####
t_steps = 0.0f0:0.01f0:1.0f0;

SI_prior_samples = euler_maruyama(
    base_sample,
    t_steps,
    num_samples,
    drift_term;
    # diffusion_term = t -> 10.0f0 .* gamma(t),
    # prior_score_term = prior_score_term
);
SI_prior_samples = SI_prior_samples[:, end, :];

SI_prior_pdf = get_kde_pdf(SI_prior_samples);
SI_prior_pdf_diagonal = get_pdf_diagonal(SI_prior_pdf);

Plots.plot(
    plot_pdf(SI_prior_pdf, "SI prior pdf"),
    plot_pdf(x1_pdf, "x1 pdf"),
    plot_pdf(likelihood_pdf, "likelihood pdf"),
    plot_pdf(posterior_pdf, "posterior pdf"),
    Plots.plot(
        [
            x1_pdf_diagonal,
            likelihood_pdf_diagonal,
            SI_prior_pdf_diagonal,
            posterior_pdf_diagonal
        ],
        linewidth = 5,
        label = ["x1 pdf" "likelihood pdf" "SI prior pdf" "posterior pdf"]
    ),
    layout = (3, 2),
    size = (1200, 800)
)

##### Stochastic Interpolant Posterior #####
function post_likelihood_function(x, obs, cov_inv)
    # jac_Hxt = Zygote.jacobian(x -> obs_operator(x), x)[1]
    jac_Hxt = H
    out = cov_inv * (obs - obs_operator(x))
    return transpose(jac_Hxt) * out
end;

function posterior_score_term(x, x0, drift, t; y = obs)
    if beta(t) < 1.0f-8
        return zeros(Float32, length(x))
    end

    likelihood_score = zeros(Float32, length(x))
    obs_interpolant = alpha(t) .* obs_operator(x0) .+ beta(t) .* y #.+ gamma(t) .* sqrt.(t) .* randn(Float32, size(x))

    interpolant_cov = beta(t) .^ 2 .* obs_cov + gamma(t) .^ 2 .* t .* H * transpose(H)
    interpolant_cov_inv = inv(interpolant_cov)

    # Evaluate likelihood score for each sample
    likelihood_score = zeros(Float32, size(x))
    for i in 1:size(x, 2)
        likelihood_score[:, i] =
            post_likelihood_function(x[:, i], obs_interpolant[:, i], interpolant_cov_inv)
    end

    return likelihood_score
end;

t_steps = 0.0f0:0.001f0:1.0f0;
SI_posterior_samples = euler_maruyama(
    base_sample,
    t_steps,
    num_samples,
    drift_term;
    diffusion_term = t -> 10.5f0 .* gamma(t),
    prior_score_term = prior_score_term,
    likelihood_score_term = posterior_score_term
);
SI_posterior_samples = SI_posterior_samples[:, end, :];

SI_posterior_pdf = get_kde_pdf(SI_posterior_samples);
SI_posterior_pdf_diagonal = get_pdf_diagonal(SI_posterior_pdf);

Plots.plot(
    plot_pdf(SI_posterior_pdf, "SI posterior pdf"),
    plot_pdf(posterior_pdf, "posterior pdf"),
    plot_pdf(x1_pdf, "x1 pdf"),
    plot_pdf(likelihood_pdf, "likelihood pdf"),
    Plots.plot(
        [
            x1_pdf_diagonal,
            likelihood_pdf_diagonal,
            SI_posterior_pdf_diagonal,
            posterior_pdf_diagonal
        ],
        linewidth = 5,
        label = ["x1 pdf" "likelihood pdf" "SI posterior pdf" "posterior pdf"]
    ),
    layout = (3, 2),
    size = (1200, 1200)
)
