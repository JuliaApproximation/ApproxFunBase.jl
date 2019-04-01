using ApproxFunBase, Test
import ApproxFunBase: PointSpace, HeavisideSpace, PiecewiseSegment

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
end