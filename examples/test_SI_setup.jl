import SciGenML
import SciGenML.NeuralNetworkArchitectures as NNArchitectures
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Config as Config
import Lux
import Configurations
import Distributions
import Plots

NUM_SAMPLES = 1000

config = Configurations.from_toml(Config.Hyperparameters, "configs/dense_SI.toml")

drift_model = NNArchitectures.DenseNeuralNetwork(
    config.architecture.in_features,
    config.architecture.out_features,
    config.architecture.hidden_features;
);

model =
    Models.StochasticInterpolantGenerativeModel(config.model.interpolant_type, drift_model);

rng = Lux.Random.default_rng()
x_data_dist = Distributions.Normal(0.0, 1.0)
x_data = rand(rng, x_data_dist, NUM_SAMPLES)

y_data_dist = Distributions.MixtureModel(Distributions.Normal[
    Distributions.Normal(-3.0, 1.0),
    Distributions.Normal(3.0, 1.0)
])

y_data = rand(rng, y_data_dist, NUM_SAMPLES)

Plots.histogram(x_data, bins = 50, normalize = :density, label = "x_data")
Plots.histogram!(y_data, bins = 100, normalize = :density, label = "y_data")

y, st = model((x_data, y_data), ps, st);

const loss_fun = Lux.MSELoss()

mse = loss_fun(y, y_data)

(ps, st) = Training.simple_train(;
    model = model,
    ps = ps,
    st = st,
    data = (x = x_data, y = y_data)
)

y_pred, st = model(x_data, ps, st);
mse = loss_fun(y_pred, y_data)
