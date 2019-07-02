using ApproxFunBase, Test
import ApproxFunBase: PointSpace, HeavisideSpace, PiecewiseSegment, dimension, Vec, checkpoints

@testset "Spaces" begin
    @testset "PointSpace" begin
        @test eltype(domain(PointSpace([0,0.1,1])) ) == Float64

        f=Fun(x->(x-0.1),PointSpace([0,0.1,1]))
        @test roots(f) == [0.1]

        a=Fun(exp,space(f))
        @test f/a == Fun(x->(x-0.1)*exp(-x),space(f))

        f = Fun(space(f),[1.,2.,3.])
    end

    @testset "Derivative operator for HeavisideSpace" begin
        H = HeavisideSpace([-1.0,0.0,1.0])
        @test Fun(H, [1.0])(1.0) == 0.0
        @test Fun(H, [0.0,1.0])(1.0) == 1.0

        H=HeavisideSpace([1,2,3])
        D=Derivative(H)
        @test domain(D)==PiecewiseSegment([1,2,3])
        @test D[1,1]==-1
        @test D[1,2]==1

        H=HeavisideSpace([1,2,3,Inf])
        D=Derivative(H)
        @test domain(D)==PiecewiseSegment([1,2,3,Inf])
        @test D[1,1]==-1
        @test D[2,2]==-1
        @test D[1,2]==1

        S = HeavisideSpace([-1.0,0.0,1.0])
        @test Derivative(S) === Derivative(S,1)

        a = HeavisideSpace(0:0.25:1)
        @test dimension(a) == 4
        @test @inferred(points(a)) == 0.125:0.25:0.875
    end

    @testset "DiracDelta integration and differentiation" begin
        δ = DiracDelta()
        h = integrate(δ)
        @test domain(h) == PiecewiseSegment([0,Inf])
        @test h(-2) == 0
        @test h(2) == 1

        δ = 0.3DiracDelta(0.1) + 3DiracDelta(2.3)
        h = integrate(δ)
        @test domain(h) == PiecewiseSegment([0.1,2.3,Inf])
        @test h(-2) == 0
        @test h(2) == 0.3
        @test h(3) == 3.3

        δ = (0.3+1im)DiracDelta(0.1) + 3DiracDelta(2.3)
        h = integrate(δ)
        @test domain(h) == PiecewiseSegment([0.1,2.3,Inf])
        @test h(-2) == 0
        @test h(2) == 0.3+1im
        @test h(3) == 3.3+1im
    end

    @testset "Multivariate" begin
        a = HeavisideSpace(0:0.25:1)
        @test @inferred(dimension(a^2)) == dimension(a)^2
        @test @inferred(domain(a^2)) == domain(a)^2
        @test @inferred(points(a^2)) == vec(Vec.(points(a), points(a)'))
        @test  @inferred(checkpoints(a^2)) == vec(Vec.(checkpoints(a)', checkpoints(a)))
        
        aa2 = TensorSpace(a , a^2)
        @test dimension(aa2) == dimension(a)^3
        @test @inferred(domain(aa2)) == domain(a)^3
        @test @inferred(points(aa2)) == vec(Vec.(points(a), points(a)', reshape(points(a), 1,1,4)))
        @test  @inferred(checkpoints(aa2)) == vec(Vec.(reshape(checkpoints(a), 1,1,length(checkpoints(a))), checkpoints(a)', checkpoints(a)))

        @test dimension(a^3) == dimension(a)^3
        @test @inferred(domain(a^3)) == domain(a)^3
        @test_broken @inferred(points(a^3)) == vec(Vec.(points(a), points(a)', reshape(points(a), 1,1,4)))
    end
end