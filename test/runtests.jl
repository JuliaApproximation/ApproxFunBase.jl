using ApproxFunBase, ContinuumArrays, OrthogonalPolynomialsQuasi, LinearAlgebra, Random, Test
import ApproxFunBase: Infinity, ∞

@testset "Spline" begin
    L = LinearSpline(range(0,1; length=1000))
    f = Fun(exp, L)
    @test f(0.1) ≈ exp(0.1) atol=1E-2
    @test_throws BoundsError f(1.1)
    @test space(f') isa HeavisideSpline
    @test f'(0.1) ≈ exp(0.1) atol=1E-2

    @test (f+f)(0.1) ≈ 2exp(0.1) atol=1E-2
    @test (2f)(0.1) ≈ 2exp(0.1) atol=1E-2
end

@testset "Chebyshev" begin
    f = Fun(exp, Chebyshev())
    @test f(0.1) ≈ exp(0.1)
    @test f'(0.1) ≈ exp(0.1)
    @test space(f') isa ChebyshevU
    @test f''(0.1) ≈ exp(0.1)
    @test space(f'') == Ultraspherical(2)

    @test (f+f)(0.1) ≈ 2exp(0.1)
    @test (2f)(0.1) ≈ 2exp(0.1)
end


@testset "Domain" begin
    @test 0.45-0.65im ∉ Segment(-1,1)

    @test ApproxFunBase.AnySegment() == ApproxFunBase.AnySegment()

    @test ApproxFunBase.dimension(ChebyshevInterval()) == 1
    @test ApproxFunBase.dimension(ChebyshevInterval()^2) == 2
    @test ApproxFunBase.dimension(ChebyshevInterval()^3) == 3

    @test isambiguous(convert(ApproxFunBase.Point,ApproxFunBase.AnyDomain()))
    @test isambiguous(ApproxFunBase.Point(ApproxFunBase.AnyDomain()))

    @test_skip ApproxFunBase.Point(NaN) == ApproxFunBase.Point(NaN)
end

@time include("MatrixTest.jl")
@time include("SpacesTest.jl")


@testset "blockbandwidths for FiniteOperator of pointscompatibleace bug" begin
    S = ApproxFunBase.PointSpace([1.0,2.0])
    @test ApproxFunBase.blockbandwidths(FiniteOperator([1 2; 3 4],S,S)) == (0,0)
end

@testset "DiracDelta sampling" begin
    δ = 0.3DiracDelta(0.1) + 3DiracDelta(2.3)
    Random.seed!(0)
    for _=1:10
        @test sample(δ) ∈ [0.1, 2.3]
    end
    Random.seed!(0)
    r = sample(δ, 10_000)
    @test count(i -> i == 0.1, r)/length(r) ≈ 0.3/(3.3) atol=0.01
end

@time include("ETDRK4Test.jl")
