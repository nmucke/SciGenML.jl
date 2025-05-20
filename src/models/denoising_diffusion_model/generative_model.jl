"""
    DenoisingDiffusionModel

A denoising diffusion model generative model.

The denoising diffusion model generative model is a generative model that uses a denoising diffusion model approach to generate data.

It is a special case of the stochastic interpolant generative model where there is no
noise in the interpolant.
"""
mutable struct DenoisingDiffusionModel <: Models.GenerativeModel
    interpolant_coefs::Any
    denoiser::Any
    ps::Any
    st::Any
    trait::Any
    device::Any

    function DenoisingDiffusionModel(denoiser,)
        denoiser_ps, denoiser_st = Lux.setup(Lux.Random.default_rng(), denoiser)

        ps = (; denoiser = denoiser_ps)
        st = (; denoiser = denoiser_st)
        return new(
            linear_interpolant_coefs(Models.Deterministic()),
            denoiser,
            ps,
            st,
            Models.Deterministic(),
            DEFAULT_DEVICE
        )
    end

    # Constructor with interpolant type
    function DenoisingDiffusionModel(interpolant_type::String, denoiser)
        denoiser_ps, denoiser_st = Lux.setup(Lux.Random.default_rng(), denoiser)

        ps = (; denoiser = denoiser_ps)
        st = (; denoiser = denoiser_st)
        return new(
            get_interpolant_coefs(Models.Deterministic(), interpolant_type),
            denoiser,
            ps,
            st,
            Models.Deterministic(),
            DEFAULT_DEVICE
        )
    end

    ### Constructor from config
    function DenoisingDiffusionModel(config::Config.Hyperparameters,)

        # Define velocity model
        denoiser_model = Architectures.DenseNeuralNetwork(
            config.architecture.in_features,
            config.architecture.out_features,
            config.architecture.hidden_features;
        );
        return DenoisingDiffusionModel(config.model.interpolant_type, denoiser_model)
    end
end
