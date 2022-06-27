
@testset "some functions" begin
    A = randn(rng, 2, 3, 4)
    @test reverse(size(A)) == size(JJ.reversedims(A))

    @test JJ.insert(.+, A) ≈ sum(A; dims=ndims(A))
    @test JJ.rank"x -> JJ.insert(*, x) 1"(A) ≈ dropdims(prod(A; dims=1); dims=1)

    @test JJ.insert(+, [1 2 3; 4 5 6]) == [6, 15]
    @test JJ.insert(-, [1, 2, 3]) == 2
end

@testset "table" begin
    @test JJ.table(*, 1:3, 1:4) == reshape(1:3, 3, 1) .* reshape(1:4, 1, 4)
    @test size(JJ.table((x,y) -> [x, y], ones(3, 4), zeros(5))) == (2, 3, 4, 5)

    A = randn(rng, 10, 20)
    B = randn(rng, 20, 30)
    @test JJ.table(JJ.rank"1 (x, y) -> sum(x .* y) 1", A', B) ≈ A * B
    @test JJ.insert(+, JJ.rank"1 (x, y) -> JJ.table(.*, x, y) 1"(A, B')) ≈ A * B
    
    # also test some variants with scalar cases
    v = randn(rng, 10)
    u = randn(rng, 20)
    @test JJ.table(JJ.rank"1 (x, y) -> sum(x .* y) 1", A, v) ≈ (v' * A)'
    @test JJ.table(JJ.rank"1 (x, y) -> sum(x .* y) 1", u, .- u) ≈ - u' * u
    @test JJ.table(JJ.rank"1 (x, y) -> sum(x .* y) 0", A, u) ≈ [sum(A[:, i] .* u[j]) for i in axes(A, 2), j in axes(u, 1)]
    @test JJ.table(*, A, 2.0) ≈ 2.0 * A
    @test JJ.table(JJ.rank"1 ./ 1", 3.0, A) ≈ 3.0 ./ A
    @test JJ.table(*, 2, 3) == 6
end
