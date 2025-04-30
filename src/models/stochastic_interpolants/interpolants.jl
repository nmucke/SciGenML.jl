"""
    InterpolantCoefs

    A struct that contains the alpha and beta function coefficients and their derivatives.

    The interpolant is defined as:
    x(t) = α(t) * x0 + β(t) * x1

    The derivatives are:
    dx/dt = α'(t) * x0 + β'(t) * x1
"""
struct InterpolantCoefs
    alpha::Function
    beta::Function
    alpha_diff::Function
    beta_diff::Function
end

"""
    linear_interpolant

    Returns the linear interpolant.

    The linear interpolant is defined as:
    α(t) = 1 - t
    β(t) = t

    The derivatives are:
    α'(t) = -1
    β'(t) = 1
"""
function linear_interpolant_coefs()
    alpha = t -> 1.0f0 .- t
    beta = t -> t

    alpha_diff = t -> -1.0f0
    beta_diff = t -> 1.0f0

    return InterpolantCoefs(alpha, beta, alpha_diff, beta_diff)
end

"""
    quadratic_interpolant

    Returns the quadratic interpolant.

    The quadratic interpolant is defined as:
    α(t) = 1 - t
    β(t) = t^2

    The derivatives are:
    α'(t) = -1
    β'(t) = 2t
"""
function quadratic_interpolant_coefs()
    alpha = t -> 1.0f0 .- t
    beta = t -> t .^ 2

    alpha_diff = t -> -1.0f0
    beta_diff = t -> 2.0f0 * t

    return InterpolantCoefs(alpha, beta, alpha_diff, beta_diff)
end

"""
    compute_interpolant

    Computes the interpolant at a given time.

    Args:
        x0: The starting point.
        x1: The ending point.
        interpolant: The interpolant.
        t: The time.
"""
function compute_interpolant(x0, x1, interpolant_coefs::InterpolantCoefs, t)
    return interpolant_coefs.alpha(t) .* x0 .+ interpolant_coefs.beta(t) .* x1
end

"""
    compute_interpolant_diff

    Computes the derivative of the interpolant at a given time.

    Args:
        x0: The starting point.
        x1: The ending point.
        interpolant: The interpolant.
        t: The time.
"""
function compute_interpolant_diff(x0, x1, interpolant_coefs::InterpolantCoefs, t)
    return interpolant_coefs.alpha_diff(t) * x0 + interpolant_coefs.beta_diff(t) * x1
end
