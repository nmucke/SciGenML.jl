
@testitem "Dense Architecture" begin
    import SciGenML.Architectures as Architectures
    import SciGenML.Training as Training
    using Lux: Lux
    using Optimisers: Optimisers
    using Zygote

    IN_FEATURES = 10
    OUT_FEATURES = 1
    HIDDEN_FEATURES = [10, 10]
    ACTIVATION_FUNCTION = x -> Lux.relu(x)
    BATCH_SIZE = 5
    @testset "Forward pass Test 1" begin

        # Test basic initialization
        model = Architectures.DenseNeuralNetwork(
            IN_FEATURES,
            OUT_FEATURES,
            HIDDEN_FEATURES;
            activation_function = ACTIVATION_FUNCTION
        )

        # Test model setup
        rng = Lux.Random.default_rng()
        ps, st = Lux.setup(rng, model)

        @test ps isa NamedTuple
        @test st isa NamedTuple

        @test size(ps.input_layer.weight) == (HIDDEN_FEATURES[1], IN_FEATURES)
        @test size(ps.input_layer.bias) == (HIDDEN_FEATURES[1],)
        @test size(ps.hidden_layers[1].weight) == (HIDDEN_FEATURES[2], HIDDEN_FEATURES[1])
        @test size(ps.hidden_layers[1].bias) == (HIDDEN_FEATURES[2],)
        @test size(ps.output_layer.weight) == (OUT_FEATURES, HIDDEN_FEATURES[end])
        @test !haskey(ps.output_layer, :bias)

        # Test forward pass
        x = rand(rng, DEFAULT_TYPE, IN_FEATURES, BATCH_SIZE)

        _st = Lux.testmode(st)
        y, st = model(x, ps, _st)

        @test y isa AbstractArray
        @test size(y) == (OUT_FEATURES, BATCH_SIZE)
    end

    IN_FEATURES = 2
    OUT_FEATURES = 32
    HIDDEN_FEATURES = [16, 16, 16]
    ACTIVATION_FUNCTION = x -> Lux.sigmoid(x)
    BATCH_SIZE = 5
    @testset "Forward pass Test 2" begin
        model = Architectures.DenseNeuralNetwork(
            IN_FEATURES,
            OUT_FEATURES,
            HIDDEN_FEATURES;
            activation_function = ACTIVATION_FUNCTION
        )

        rng = Lux.Random.default_rng()
        ps, st = Lux.setup(rng, model)

        @test ps isa NamedTuple
        @test st isa NamedTuple

        @test size(ps.input_layer.weight) == (HIDDEN_FEATURES[1], IN_FEATURES)
        @test size(ps.input_layer.bias) == (HIDDEN_FEATURES[1],)
        @test size(ps.hidden_layers[1].weight) == (HIDDEN_FEATURES[2], HIDDEN_FEATURES[1])
        @test size(ps.hidden_layers[1].bias) == (HIDDEN_FEATURES[2],)
        @test size(ps.hidden_layers[2].weight) == (HIDDEN_FEATURES[3], HIDDEN_FEATURES[2])
        @test size(ps.hidden_layers[2].bias) == (HIDDEN_FEATURES[3],)
        @test size(ps.output_layer.weight) == (OUT_FEATURES, HIDDEN_FEATURES[end])
        @test !haskey(ps.output_layer, :bias)

        x_batch = rand(rng, DEFAULT_TYPE, IN_FEATURES, BATCH_SIZE)

        _st = Lux.testmode(st)
        y_batch, st = model(x_batch, ps, _st)

        @test y_batch isa AbstractArray
        @test size(y_batch) == (OUT_FEATURES, BATCH_SIZE)
    end

    IN_FEATURES = 10
    OUT_FEATURES = 1
    HIDDEN_FEATURES = [10, 10]
    ACTIVATION_FUNCTION = x -> Lux.relu(x)
    @testset "Training Test" begin
        model = Architectures.DenseNeuralNetwork(
            IN_FEATURES,
            OUT_FEATURES,
            HIDDEN_FEATURES;
            activation_function = ACTIVATION_FUNCTION
        )

        rng = Lux.Random.default_rng()
        ps, st = Lux.setup(rng, model)

        x_data = rand(rng, DEFAULT_TYPE, IN_FEATURES, 100)
        y_data = rand(rng, DEFAULT_TYPE, OUT_FEATURES, 100)

        loss_fn = Lux.MSELoss()

        _st = Lux.testmode(st)
        y_pred, st = model(x_data, ps, _st)
        init_loss = loss_fn(y_pred, y_data)

        st = Lux.trainmode(st)
        ps, st = Training.simple_train(;
            model = model,
            ps = ps,
            st = st,
            data = (x = x_data, y = y_data),
            verbose = false
        )

        _st = Lux.testmode(st)
        y_pred, st = model(x_data, ps, _st)
        final_loss = loss_fn(y_pred, y_data)

        @test final_loss < init_loss
    end

    IN_FEATURES = 10
    OUT_FEATURES = 1
    HIDDEN_FEATURES = [10, 10]
    ACTIVATION_FUNCTION = x -> Lux.relu(x)
    CONDITIONING_FEATURES = 10
    BATCH_SIZE = 5
    @testset "Conditional forward pass" begin
        model = Architectures.DenseNeuralNetwork(
            (IN_FEATURES, CONDITIONING_FEATURES),
            OUT_FEATURES,
            HIDDEN_FEATURES;
            activation_function = ACTIVATION_FUNCTION
        )

        rng = Lux.Random.default_rng()
        ps, st = Lux.setup(rng, model)

        x_data = rand(rng, DEFAULT_TYPE, IN_FEATURES, BATCH_SIZE)
        c_data = rand(rng, DEFAULT_TYPE, CONDITIONING_FEATURES, BATCH_SIZE)
        y_data = rand(rng, DEFAULT_TYPE, OUT_FEATURES, BATCH_SIZE)

        _st = Lux.testmode(st)
        y, st = model((x_data, c_data), ps, _st)

        @test y isa AbstractArray
        @test size(y) == (OUT_FEATURES, BATCH_SIZE)
    end
end
