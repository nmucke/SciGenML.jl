
SUPPORTED_INTERPOLANT_TYPES = ["linear", "quadratic"]

"""
    get_interpolant_coefs(type::String)

    Get the interpolant coefficients for a given type. 
    Supported types are:
    - "linear"
    - "quadratic"
"""
function get_interpolant_coefs(
    trait::Union{Models.Deterministic, Models.Stochastic},
    type::String
)
    if !(type in SUPPORTED_INTERPOLANT_TYPES)
        throw(ArgumentError("Unsupported interpolant type: $type. Supported types are: $(SUPPORTED_INTERPOLANT_TYPES)"))
    end
    if type == "linear"
        return linear_interpolant_coefs(trait)
    elseif type == "quadratic"
        return quadratic_interpolant_coefs(trait)
    elseif type == "diffusion"
        return diffusion_interpolant_coefs(trait)
    end
end
