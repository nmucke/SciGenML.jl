using Test
using TestItemRunner

@testitem "Main Module Loading" begin
    using SciGenML
    @test isdefined(Main, :SciGenML)
end


using TestItemRunner
using SciGenML
using Test
@testitem "SciGenML Loaded" begin
    @test isdefined(:SciGenML)
end

isdefined(:SciGenML)

x = 2

using Pkg
Pkg.test("SciGenML")
