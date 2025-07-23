include("config.jl")

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
