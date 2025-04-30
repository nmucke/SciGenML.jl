using SciGenML
import SciGenML.Architectures as Architectures
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Config as Config
import Lux
import Configurations
import Distributions
import Plots

NUM_SAMPLES = 100000

# Load config
config = Configurations.from_toml(Config.Hyperparameters, "configs/dense_SI.toml");

# Define drift model
drift_model = Architectures.DenseNeuralNetwork(
    config.architecture.in_features,
    config.architecture.out_features,
    config.architecture.hidden_features;
);

# Define generative model
model =
    Models.StochasticInterpolantGenerativeModel(config.model.interpolant_type, drift_model);

# Define data distributions
rng = Lux.Random.default_rng();
x_data_dist = Distributions.Normal(0.0, 1.0);
x_data = rand(rng, x_data_dist, (1, NUM_SAMPLES)) .|> DEFAULT_TYPE;

y_data_dist = Distributions.MixtureModel(Distributions.Normal[
    Distributions.Normal(-3.0, 1.0),
    Distributions.Normal(3.0, 1.0)
]);

y_data = rand(rng, y_data_dist, (1, NUM_SAMPLES)) .|> DEFAULT_TYPE;

Plots.histogram(x_data[1, :], bins = 50, normalize = :density, label = "x_data");
Plots.histogram!(y_data[1, :], bins = 100, normalize = :density, label = "y_data");

# Train model
model = Training.train(model, (base = x_data, target = y_data), config; verbose = true);
