"""
    drift_term(x, x0, t)

Compute the drift term of the stochastic interpolant.
"""
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

"""
    eval_drift_term(x, x0, t)

Evaluate the drift term of the stochastic interpolant.
"""
function eval_drift_term(x, x0, t)
    n_samples = size(x, 2)
    out = zeros(Float32, 2, n_samples)
    for i in 1:n_samples
        out[:, i] = drift_term(x[:, i], x0[:, i], t)
    end
    return out
end;

"""
    prior_score_term(x, x0, drift, t)

Compute the prior score term of the stochastic interpolant.
"""
function prior_score_term(x, x0, drift, t)
    A = 1.0f0 ./ (t .* gamma(t) .* (beta_diff(t) .* gamma(t) .- beta(t) .* gamma_diff(t)))
    c = beta_diff(t) .* x .+ (beta(t) .* alpha_diff(t) - alpha(t) .* beta_diff(t)) .* x0
    return A .* (beta(t) .* drift - c)
end;

"""
    euler_maruyama(
        x0, 
        t_steps, 
        num_samples, 
        drift_term; 
        diffusion_term = gamma
    )

Euler-Maruyama method for the stochastic interpolant.
"""
function euler_maruyama(x0, t_steps, num_samples, drift_term; diffusion_term = gamma)
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

function euler_maruyama(
    x0,
    t_steps,
    num_samples,
    drift_term;
    diffusion_term = gamma,
    prior_score_term = prior_score_term
)
    x = zeros(Float32, 2, length(t_steps), num_samples)
    dt = t_steps[2] - t_steps[1]

    if length(size(x0)) == 1
        x0 = repeat(x0, 1, num_samples)
    end
    x[:, 1, :] = x0

    z = Random.randn(Float32, 2, num_samples)
    x_new = x[:, 1, :] .+ drift_term(x[:, 1, :], x0, t_steps[1]) .* dt
    x_new = x_new .+ sqrt(dt) .* gamma(t_steps[1]) .* z
    x[:, 2, :] = x_new

    for i in 3:length(t_steps)
        z = Random.randn(Float32, 2, num_samples)
        diffusion_term_val = diffusion_term(t_steps[i - 1])

        drift = drift_term(x_new, x0, t_steps[i - 1])
        prior_score = prior_score_term(x_new, x0, drift, t_steps[i - 1])

        drift =
            drift .+
            0.5f0 .* (diffusion_term_val .^ 2 .- gamma(t_steps[i - 1]) .^ 2) .* prior_score

        x_new = x_new .+ drift .* dt
        x_new = x_new .+ sqrt(dt) .* diffusion_term_val .* z
        x[:, i, :] = x_new
    end
    return x
end

function likelihood_score_term(x, x0, drift, t)
    return zeros(Float32, 2)
end

function euler_maruyama(
    x0,
    t_steps,
    num_samples,
    drift_term;
    diffusion_term = gamma,
    prior_score_term = prior_score_term,
    likelihood_score_term = likelihood_score_term
)
    x = zeros(Float32, 2, length(t_steps), num_samples)
    dt = t_steps[2] - t_steps[1]

    if length(size(x0)) == 1
        x0 = repeat(x0, 1, num_samples)
    end
    x[:, 1, :] = x0

    z = Random.randn(Float32, 2, num_samples)
    x_new = x[:, 1, :] .+ drift_term(x[:, 1, :], x0, t_steps[1]) .* dt
    x_new = x_new .+ sqrt(dt) .* gamma(t_steps[1]) .* z
    x[:, 2, :] = x_new

    for i in 3:length(t_steps)
        z = Random.randn(Float32, 2, num_samples)
        diffusion_term_val = diffusion_term(t_steps[i - 1])

        drift = drift_term(x_new, x0, t_steps[i - 1])
        prior_score = prior_score_term(x_new, x0, drift, t_steps[i - 1])
        likelihood_score = likelihood_score_term(x_new, x0, drift, t_steps[i - 1])

        drift =
            drift .+
            0.5f0 .* (diffusion_term_val .^ 2 .- gamma(t_steps[i - 1]) .^ 2) .* prior_score

        drift =
            drift .+ # 0.5f0 .* diffusion_term_val .^ 2 .* likelihood_score
            0.5f0 .* (diffusion_term_val .^ 2 .- gamma(t_steps[i - 1]) .^ 2) .*
            likelihood_score

        x_new = x_new .+ drift .* dt
        x_new = x_new .+ sqrt(dt) .* diffusion_term_val .* z
        x[:, i, :] = x_new
    end
    return x
end
