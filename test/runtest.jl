
using JJ

using Test
using Random

rng = MersenneTwister(1234)

@testset "ranking" begin
    include("test_ranked.jl")
end

@testset "jfuns" begin
    include("test_jfun.jl")
end
