using ApproxFunBase
using Test
using ApproxFunBase: Block


@testset "RaggedMatrix" begin
    cols=Int[rand(1:k+2) for k=1:5]
    B=ApproxFunBase.rrand(Float64,maximum(cols),cols)
    cols=Int[rand(1:k+2) for k=1:size(B,1)]
    A=ApproxFunBase.rrand(Float64,maximum(cols),cols)
    @test Matrix(A)*Matrix(B) ≈ Matrix(A*B)

    @test ApproxFunBase.RaggedMatrix(B) === B
    @test ApproxFunBase.RaggedMatrix{Float64}(B) === B
    @test Matrix(ApproxFunBase.RaggedMatrix{ComplexF64}(B)) == Matrix{ComplexF64}(Matrix(B))

    B = ApproxFunBase.brand(10,10,2,3)
    @test Matrix(B) == Matrix(ApproxFunBase.RaggedMatrix(B))
    @test ApproxFunBase.RaggedMatrix(B) == ApproxFunBase.RaggedMatrix{Float64}(B)
    @test ApproxFunBase.RaggedMatrix(ApproxFunBase.BandedMatrix{ComplexF64}(B)) == ApproxFunBase.RaggedMatrix{ComplexF64}(B)

    @testset "similar" begin
        B = ApproxFunBase.brand(10,10,2,3)
        R = ApproxFunBase.RaggedMatrix(B)
        S = similar(R, ComplexF64)
        @test S isa ApproxFunBase.RaggedMatrix
        @test eltype(S) == ComplexF64
        @test size(S) == size(R)
    end
end
