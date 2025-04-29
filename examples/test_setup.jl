using SciGenML: SciGenML
import SciGenML.NeuralNetworkArchitectures as NNArchitectures
import SciGenML.Training as Training
using Lux: Lux
using Random
using Optimisers
using Zygote

model = NNArchitectures.DenseNN(;
    in_features = 10,
    out_features = 1,
    hidden_features = [10, 10],
    activation_function = x -> Lux.relu(x)
);

ps, st = Lux.setup(Lux.Random.default_rng(), model);

rng = Lux.Random.default_rng()
x_data = rand(rng, Float32, 10, 100)
y_data = rand(rng, Float32, 1, 100)

y, st = model(x_data, ps, st);

const loss_fun = Lux.MSELoss()

mse = loss_fun(y, y_data)

ps,
st = Training.simple_train(;
    model = model,
    ps = ps,
    st = st,
    data = (x = x_data, y = y_data)
)

y_pred, st = model(x_data, ps, st);
mse = loss_fun(y_pred, y_data)

# # Both these steps can be combined into a single call
# gs, loss, stats, train_state = Training.single_train_step!(
#     AutoZygote(), compute_loss,
#     (x_data, y_data), train_state
# )
