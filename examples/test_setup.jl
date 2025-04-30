import SciGenML
import SciGenML.NeuralNetworkArchitectures as NNArchitectures
import SciGenML.Training as Training
import SciGenML.Config as Config
import Lux
import Configurations

config =
    Configurations.from_toml(Config.Hyperparameters, "configs/dense_neural_network.toml")

model = NNArchitectures.DenseNeuralNetwork(
    config.architecture.in_features,
    config.architecture.out_features,
    config.architecture.hidden_features;
);

ps, st = Lux.setup(Lux.Random.default_rng(), model);

rng = Lux.Random.default_rng()
x_data = rand(rng, Float32, 20, 100)
c_data = rand(rng, Float32, 10, 100)
y_data = rand(rng, Float32, 1, 100)

y, st = model((x_data, c_data), ps, st);

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
