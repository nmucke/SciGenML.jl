using Test


@testitem "Main Module Loading" begin
    using SciGenML
    @test isdefined(Main, :SciGenML)
end

@testitem "Submodule Loading" begin
    @test isdefined(SciGenML, :Models)
    @test isdefined(SciGenML, :NeuralNetworkArchitectures)
    @test isdefined(SciGenML, :NeuralNetworkLayers)
    @test isdefined(SciGenML, :TimeIntegrators)
    @test isdefined(SciGenML, :Data)
end

