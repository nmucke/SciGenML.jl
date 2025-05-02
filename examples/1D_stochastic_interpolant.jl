using SciGenML
import SciGenML.Architectures as Architectures
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Sampling as Sampling
import SciGenML.Config as Config
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators
import Lux
import Configurations
import Distributions
import Plots
import Random

NUM_SAMPLES = 10000
rng = Lux.Random.default_rng();

##### Load config #####
config = Configurations.from_toml(Config.Hyperparameters, "configs/dense_SI.toml");

##### Define generative model #####
# Define velocity model
# velocity_model = Architectures.DenseNeuralNetwork(
#     config.architecture.in_features,
#     config.architecture.out_features,
#     config.architecture.hidden_features;
# );

# # Define score model
# score_model = Architectures.DenseNeuralNetwork(
#     config.architecture.in_features,
#     config.architecture.out_features,
#     config.architecture.hidden_features;
# );

# Define generative model
# SI_model = Models.StochasticInterpolant(velocity_model, score_model);
# SI_model = Models.StochasticInterpolant(velocity_model, );
SI_model = Models.StochasticInterpolant(config,);

##### Define training data #####
x_data_dist = Distributions.Normal(0.0, 1.0);
x_data = rand(rng, x_data_dist, (1, NUM_SAMPLES)) .|> DEFAULT_TYPE;

y_data_dist = Distributions.MixtureModel(Distributions.Normal[
    Distributions.Normal(-3.0, 1.0),
    Distributions.Normal(3.0, 1.0)
]);
y_data = rand(rng, y_data_dist, (1, NUM_SAMPLES)) .|> DEFAULT_TYPE;

Plots.histogram(x_data[1, :], bins = 100, normalize = :density, label = "x_data")
Plots.histogram!(y_data[1, :], bins = 100, normalize = :density, label = "y_data")

##### Train model #####
data = (base = x_data, target = y_data);
model = Training.train(SI_model, data, config; verbose = true);

##### Sample using model #####
si_samples = Sampling.sample(
    Models.Deterministic(),
    model,
    20;
    num_samples = NUM_SAMPLES,
    verbose = true,
    stepper = TimeIntegrators.RK4_step
);

Plots.histogram(si_samples[1, :], bins = 100, normalize = :density, label = "SI samples")
Plots.histogram!(y_data[1, :], bins = 100, normalize = :density, label = "Target samples")
