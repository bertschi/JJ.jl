
using Test

@testset "julienned backend" begin
    A = reshape(1:6, 3, 2)
    @test JJ.enframe(A, Val(0)) === A
    @test size(JJ.enframe(A, Val(1))) == (2,)

    @test JJ.combine([[1, 2, 3], [4, 5, 6]]) == A

    B = reshape(1:24, 4, 3, 2)
    for i ∈ 0:ndims(B)
        @test JJ.combine(JJ.enframe(B, Val(i))) == B
    end
end

@testset "ranked function" begin
    @test map(size, JJ.makeagree([1, 2], [1 2 3; 4 5 6])) == ((1, 2), (2, 3))

    @test JJ.ranked(sum, Val(1))([1 2 3; 4 5 6]) == [5, 7, 9]

    @test JJ.ranked(Val(1), /, Val(0))([1 2 3; 4 5 6], [1, 2, 3]) == [1 1 1; 4 2.5 2]
    @test JJ.rank"1 / 0"([1 2 3; 4 5 6], [1, 2, 3]) == [1 1 1; 4 2.5 2]
end
