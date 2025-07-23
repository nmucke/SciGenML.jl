
##### Constants #####
Id = [1.0f0 0.0f0; 0.0f0 1.0f0];

x_range = range(-5, 5, length = 150);
y_range = range(-5, 5, length = 150);

##### Parameters #####
num_samples = 10000;

##### Base sample #####
base_sample = [1.0f0, 1.0f0];

##### Distributions #####
x0_cov = Id;
x0_dist = Dist.MvNormal([0.0f0, 0.0f0], x0_cov);

# Prior distribution
x1_mean = x -> [x[1], x[2]];
x1_cov = x -> [1.0f0 0.0f0; 0.0f0 1.0f0];
x1_cov_inv = x -> inv(x1_cov(x));
x1_dist = x -> Dist.MvNormal(x1_mean(x), x1_cov(x));

##### Observations #####
# True observation
true_x = [-1.5f0, -1.0f0];

# Observation operator
H = [1.0f0 1.5f0; 1.5f0 1.0f0];
obs_operator = x -> H * x;
obs = obs_operator(true_x);

# Observation covariance
obs_cov = [1.5f0 0.0f0; 0.0f0 1.0f0];

# Noise distribution
noise_dist = Dist.MvNormal(zeros(Float32, 2), obs_cov);

# Likelihood function
likelihood_function = (x, obs) -> Dist.pdf(noise_dist, obs - obs_operator(x))

# Likelihood distribution
likelihood_dist = x -> Dist.MvNormal(obs_operator(x), obs_cov)

##### True Posterior #####
kalman_gain =
    x1_cov(base_sample) *
    transpose(H) *
    inv(H * x1_cov(base_sample) * transpose(H) + obs_cov);
true_posterior_mean = x1_mean(base_sample) + kalman_gain * (obs - obs_operator(base_sample));
true_posterior_cov = (Id - kalman_gain * H) * x1_cov(base_sample);
true_posterior_dist = Dist.MvNormal(true_posterior_mean, true_posterior_cov);

##### Stochastic Interpolant #####
alpha = t -> 1.0f0 .- t;
beta = t -> t .^ 2;
gamma = t -> 1.0f0 .- t;

alpha_diff = t -> -1.0f0;
beta_diff = t -> 2.0f0 .* t;
gamma_diff = t -> -1.0f0;
