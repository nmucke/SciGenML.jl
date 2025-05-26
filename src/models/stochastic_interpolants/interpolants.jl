import SciGenML.Models as Models

"""
    DeterministicInterpolantCoefs

    A struct that contains the alpha and beta function coefficients and their derivatives.

    The interpolant is defined as:
    x(t) = α(t) * x0 + β(t) * x1

    The derivatives are:
    dx/dt = α'(t) * x0 + β'(t) * x1
"""
struct DeterministicInterpolantCoefs
    alpha::Function
    beta::Function
    alpha_diff::Function
    beta_diff::Function
end

"""
    StochasticInterpolantCoefs

    A struct that contains the alpha and beta function coefficients and their derivatives.

    The interpolant is defined as:
    x(t) = α(t) * x0 + β(t) * x1 + γ(t) * Z

    The derivatives are:
    dx/dt = α'(t) * x0 + β'(t) * x1 + γ'(t) * Z
"""
struct StochasticInterpolantCoefs
    alpha::Function
    beta::Function
    gamma::Function
    alpha_diff::Function
    beta_diff::Function
    gamma_diff::Function
end

"""
    linear_interpolant_coefs(trait::Models.Deterministic)

    Returns the linear interpolant.

    The linear interpolant is defined as:
    α(t) = 1 - t
    β(t) = t
    γ(t) = 1 - t, if trait == Models.Stochastic

    The derivatives are:
    α'(t) = -1
    β'(t) = 1
    γ'(t) = -1, if trait == Models.Stochastic

"""
function linear_interpolant_coefs(trait::Union{Models.Deterministic, Models.Stochastic} = Models.Stochastic())
    alpha = t -> 1.0f0 .- t
    beta = t -> t

    alpha_diff = t -> -1.0f0
    beta_diff = t -> 1.0f0

    if trait == Models.Deterministic()
        return DeterministicInterpolantCoefs(alpha, beta, alpha_diff, beta_diff)
    elseif trait == Models.Stochastic()
        gamma = t -> sqrt.(2.0f0 .* t .* (1.0f0 .- t))
        gamma_diff =
            t ->
                (1.0f0 .- 2.0f0 .* t) ./
                (sqrt(2.0f0) .* sqrt.(- (t .- 1.0f0) .* t) .+ ZERO_TOL)

        return StochasticInterpolantCoefs(
            alpha,
            beta,
            gamma,
            alpha_diff,
            beta_diff,
            gamma_diff
        )
    else
        throw(ArgumentError("Invalid trait: $trait. Only Deterministic and Stochastic are supported."))
    end
end

function linear_interpolant_coefs(::Models.Deterministic)
    alpha = t -> 1.0f0 .- t
    beta = t -> t

    alpha_diff = t -> -1.0f0
    beta_diff = t -> 1.0f0

    return DeterministicInterpolantCoefs(alpha, beta, alpha_diff, beta_diff)
end

"""
    quadratic_interpolant_coefs(trait::Models.Deterministic)

    Returns the quadratic interpolant.

    The quadratic interpolant is defined as:
    α(t) = 1 - t
    β(t) = t^2
    γ(t) = 1 - t, if trait == Models.Stochastic

    The derivatives are:
    α'(t) = -1
    β'(t) = 2t
    γ'(t) = -1, if trait == Models.Stochastic
"""
function quadratic_interpolant_coefs(trait::Union{Models.Deterministic, Models.Stochastic} = Models.Stochastic)
    alpha = t -> 1.0f0 .- t
    beta = t -> t .^ 2

    alpha_diff = t -> -1.0f0
    beta_diff = t -> 2.0f0 * t

    if trait == Models.Deterministic()
        return DeterministicInterpolantCoefs(alpha, beta, alpha_diff, beta_diff)
    elseif trait == Models.Stochastic()
        gamma = t -> alpha(t)
        gamma_diff = t -> alpha_diff(t)
        return StochasticInterpolantCoefs(
            alpha,
            beta,
            gamma,
            alpha_diff,
            beta_diff,
            gamma_diff
        )
    else
        throw(ArgumentError("Invalid trait: $trait. Only Deterministic and Stochastic are supported."))
    end
end

"""
    diffusion_interpolant_coefs(trait::Models.Deterministic)

    Returns the quadratic interpolant.

    The quadratic interpolant is defined as:
    α(t) = exp(-multiplier * t)
    β(t) = sqrt(1 - exp(-2 * multiplier * t))

    The derivatives are:
    α'(t) = -multiplier * exp(-multiplier * t)
    β'(t) = multiplier * exp(-multiplier * t) / sqrt(1 - exp(-2 * multiplier * t))
"""
function diffusion_interpolant_coefs(multiplier::Real = 1.0f0)
    # alpha = t -> exp.(-multiplier .* t)
    alpha = t -> 1.0f0 .- t
    # beta = t -> sqrt.(1.0f0 .- exp.(-2.0f0 .* multiplier .* t))
    beta = t -> t

    alpha_diff = t -> -fill!(similar(t, size(t)), 1.0f0)
    beta_diff = t -> fill!(similar(t, size(t)), 1.0f0)

    # alpha_diff = t -> -multiplier .* exp.(-multiplier .* t)
    # beta_diff = t -> 
    #         multiplier .* exp.(-2.0f0 .* multiplier .* t) ./
    #         (sqrt.(1.0f0 .- exp.(-2.0f0 .* multiplier .* t)) .+ ZERO_TOL)

    return DeterministicInterpolantCoefs(alpha, beta, alpha_diff, beta_diff)
end

"""
    Computes the deterministic interpolant at a given time.

    Args:
        x0: The starting point.
        x1: The ending point.
        t: The time.
        interpolant_coefs: The interpolant coefficients.
"""
function compute_interpolant(x0, x1, t, interpolant_coefs::DeterministicInterpolantCoefs)
    # Expand t to match dimensions of x0 except for the last dimension
    t = Utils.reshape_scalar(t, ndims(x0))

    return interpolant_coefs.alpha(t) .* x0 .+ interpolant_coefs.beta(t) .* x1
end

"""
    compute_interpolant(x0, x1, z, t, interpolant_coefs::StochasticInterpolantCoefs)

    Computes the stochastic interpolant at a given time.

    Args:
        x0: The starting point.
        x1: The ending point.
        z: The noise.
        t: The time.
        interpolant_coefs: The interpolant coefficients.
"""
function compute_interpolant(x0, x1, z, t, interpolant_coefs::StochasticInterpolantCoefs)

    # Expand t to match dimensions of x0 except for the last dimension
    t = Utils.reshape_scalar(t, ndims(x0))

    return interpolant_coefs.alpha(t) .* x0 .+ interpolant_coefs.beta(t) .* x1 .+
           interpolant_coefs.gamma(t) .* z
end

"""
    compute_interpolant_diff

    Computes the derivative of the interpolant at a given time.

    Args:
        x0: The starting point.
        x1: The ending point.
        interpolant_coefs: The interpolant coefficients.
        t: The time.
"""
function compute_interpolant_diff(
    x0,
    x1,
    t,
    interpolant_coefs::DeterministicInterpolantCoefs
)

    # Expand t to match dimensions of x0 except for the last dimension
    t = Utils.reshape_scalar(t, ndims(x0))

    return interpolant_coefs.alpha_diff(t) .* x0 .+ interpolant_coefs.beta_diff(t) .* x1
end

function compute_interpolant_diff(
    x0,
    x1,
    z,
    t,
    interpolant_coefs::StochasticInterpolantCoefs
)

    # Expand t to match dimensions of x0 except for the last dimension
    t = Utils.reshape_scalar(t, ndims(x0))

    return interpolant_coefs.alpha_diff(t) .* x0 .+ interpolant_coefs.beta_diff(t) .* x1 .+
           interpolant_coefs.gamma_diff(t) .* z
end
