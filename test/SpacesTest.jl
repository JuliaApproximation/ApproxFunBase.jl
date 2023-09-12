using ApproxFunBase
using Test
using ApproxFunBase: PointSpace, HeavisideSpace, PiecewiseSegment, dimension, SVector, checkpoints, AnyDomain
using StaticArrays
using BandedMatrices: rowrange, colrange, BandedMatrix
using LinearAlgebra

@testset "Spaces" begin
    @testset "PointSpace" begin
        @test eltype(domain(PointSpace([0,0.1,1])) ) == Float64

        f = @inferred Fun(x->(x-0.1),PointSpace([0,0.1,1]))
        @test roots(f) == [0.1]

        a = @inferred Fun(exp, space(f))
        @test f/a == @inferred Fun(x->(x-0.1)*exp(-x),space(f))

        f = @inferred Fun(space(f),[1.,2.,3.])

        @test (+f) == f

        f2 = @inferred Fun(space(f), view(Float64[1:3;], :))
        @test coefficients(f2) == coefficients(f)

        f = Fun(PointSpace(1:4), [1:4;])
        for spfn in Any[sin, cos, exp]
            @test values(spfn(f)) ≈ spfn.(points(f))
        end

        @testset "conversions" begin
            @testset for S in Any[typeof(space(f)), Any]
                T = Fun{S, Any, Any}
                fany = convert(T, f)
                @test fany isa T
                @test (@inferred oftype(f, fany)) isa typeof(f)
            end
            S = typeof(space(f))
            T = Fun{S, Any}
            fany = convert(T, f)
            @test fany isa T
            @test (@inferred oftype(f, fany)) isa typeof(f)

            # some trivial cases
            s = PointSpace(1:3)
            @test conversion_type(s, s) == s
            @test maxspace(s, s) == s
        end

        A = @inferred f * Multiplication(f)
        @test A * f == f^3

        @testset "inafunctional inference" begin
            @test @inferred !ApproxFunBase.isafunctional(Multiplication(f))
        end

        @testset "real/complex coefficients" begin
            c = [1:4;]
            for c2 in Any[c, c*im]
                g = Fun(PointSpace(1:4), c2)
                for fn in [real, imag, conj]
                    @test coefficients(fn(g)) == fn(c2)
                end
            end
        end
        @testset "intpow" begin
            sp = PointSpace(1:3)
            f = Fun(sp, Float64[1:3;])
            @test ApproxFunBase.intpow(f, 0) == f^0 == Fun(space(f), ones(ncoefficients(f)))
            for n in 1:3
                @test ApproxFunBase.intpow(f, n) == f^n == reduce(*, fill(f, n))
            end
            @test ApproxFunBase.intpow(f,-2) == f^-2 == 1/(f*f)
            @test f^2 == f^UInt(2)

            if VERSION >= v"1.8"
                @test (@inferred (x -> x^1)(sp)) == sp
                @test (@inferred (x -> x^2)(sp)) == sp * sp
                @test (@inferred (x -> x^3)(sp)) == sp * sp * sp
            else
                @test sp^1 == sp
                @test sp^2 == sp * sp
                @test sp^3 == sp * sp * sp
            end
            @test sp^4 == sp * sp * sp * sp
            @test sp^5 == sp * sp * sp * sp * sp
        end

        @testset "Fun accepts callables" begin
            struct Foo end
            (::Foo)(x) = x
            f1 = Fun(Foo(), PointSpace(1:10))
            f2 = Fun(Foo(), PointSpace(1:10), 10)
            @test coefficients(f1) == coefficients(f2)
        end

        @testset "union" begin
            @testset "trivial" begin
                s = PointSpace(0:3)
                @test union(s) == s
                @test union(s, s) == s
                @test union(s, s, s) == s
                @test union(s, s, s, s) == s
            end
        end

        f = @inferred (T -> ones(T, PointSpace(1:3)))(Float64)
        @test f == Fun(PointSpace(1:3), [1.0, 1.0, 1.0])
        f = @inferred (T -> zeros(T, PointSpace(1:3)))(Float64)
        @test space(f) == PointSpace(1:3)
        @test all(iszero(coefficients(f)))

        M = Multiplication(Fun(PointSpace(1:3), [1:3;]), PointSpace(1:3))
        @test (@inferred size(M)) == (3,3)
        M2 = M + M
        @test size(M2) == (3,3)
        M = Multiplication(Fun(PointSpace(1:3), [1:3;]))
        M2 = M + M
        infty = ApproxFunBase.InfiniteCardinal{0}()
        @test (@inferred size(M2)) == (infty, infty)

        @testset "literal pow" begin
            local f = Fun(PointSpace(1:3), Float64[1:3;])
            @test (@inferred (x -> x^0)(f)) == ApproxFunBase.intpow(f,0)
            @test (@inferred (x -> x^1)(f)) == ApproxFunBase.intpow(f,1)
            @test (@inferred (x -> x^2)(f)) == ApproxFunBase.intpow(f,2)
            @test (@inferred (x -> x^3)(f)) == ApproxFunBase.intpow(f,3)
            @test (@inferred (x -> x^4)(f)) == ApproxFunBase.intpow(f,4)

            local f = Fun(PointSpace(Float32[1:3;]), Float32[1:3;])
            g = @inferred (x -> x^0)(f)
            @test eltype(coefficients(g)) == Float32
            g = @inferred (x -> x^1)(f)
            @test eltype(coefficients(g)) == Float32
            g = @inferred (x -> x^2)(f)
            @test eltype(coefficients(g)) == Float32
        end

        @testset "static coeffs" begin
            f = Fun(PointSpace(1:3), SA[1,2,3])
            @test coefficients(f^2) == coefficients(f).^2
        end

        @testset "indexing outside bounds" begin
            f = Fun(PointSpace(1:3), Float64[1:3;])
            @test f(0) == 0
            @test f(1) == 1
            @test f(4) == 0
        end

        @testset "CachedOperator" begin
            s = PointSpace(1:3)
            Mop = Multiplication(Fun(s, 1:3))
            C = cache(Mop) : s → s
            @test C isa ApproxFunBase.CachedOperator
            CM = AbstractMatrix(C[:, :])
            @test CM == Diagonal(1:3)
            @test size(C[1:0, 1:0]) == (0, 0)
        end

        @testset "broadcast" begin
            f = Fun(PointSpace(1:3), 1:3)
            g = f^2
            h = f.^2
            for i in 1:3
                @test g(i) == h(i)
            end
        end

        @testset "euler and exp" begin
            f = Fun(PointSpace(1:3), 1:3)
            @test ℯ^f == exp(f)
            @test values(ℯ^f) == exp.(values(f))
        end

        # trivial sanity tests for PartialInverseOperator
        @testset "PartialInverseOperator" begin
            C = Conversion(PointSpace(1:3), PointSpace(1:3))
            P = PartialInverseOperator(C)
            @test AbstractMatrix(P * C) == I(size(C,1))
        end

        @testset "ConcreteOperatorFunction" begin
            A = 2I : PointSpace(1:4)
            Ainv = inv(A)
            B = convert(Operator{ComplexF64}, Ainv)
            @test B isa Operator{ComplexF64}
            @test ComplexF64.(Ainv[1:4, 1:4]) == B[1:4, 1:4]
        end

        @testset "call spaces" begin
            @test PointSpace(1:4)(3, 4) == 1
        end

        @testset "Broadcasting" begin
            f1 = x -> x
            f2 = x -> x^2
            sp = PointSpace(1:3)
            fs = Fun.((f1, f2), sp)
            @test fs[1] == Fun(f1, sp)
            @test fs[2] == Fun(f2, sp)
        end

        @testset "promote_rule" begin
            F = Fun{typeof(PointSpace(1:3)), Float32}
            @test ApproxFunBase.cfstype(F) == Float32
        end
    end

    @testset "DiracSpace" begin
        f = Fun(ApproxFunBase.DiracSpace(Float64[-1,0,1]), Float64[1,2,3])
        @test f(-5) == 0
        @test f(0.5) == 0
        @test f(5) == 0
        @test isinf(f(0))
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
        @test @inferred(points(a^2)) == vec(SVector.(points(a), points(a)'))
        @test  @inferred(checkpoints(a^2)) == vec(SVector.(checkpoints(a)', checkpoints(a)))

        aa2 = @inferred TensorSpace(a , a^2)
        @test dimension(aa2) == dimension(a)^3
        @test @inferred(domain(aa2)) == domain(a)^3
        @test @inferred(points(aa2)) == vec(SVector.(points(a), points(a)', reshape(points(a), 1,1,4)))
        @test  @inferred(checkpoints(aa2)) == vec(SVector.(reshape(checkpoints(a), 1,1,length(checkpoints(a))), checkpoints(a)', checkpoints(a)))

        @test dimension(a^3) == dimension(a)^3
        @test @inferred(domain(a^3)) == domain(a)^3
        @test_broken @inferred(points(a^3)) == vec(SVector.(points(a), points(a)', reshape(points(a), 1,1,4)))

        p = PointSpace(1:4)
        d = domain(p)
        @test domain(@inferred TensorSpace(p)) == d
        @test components(domain(@inferred TensorSpace(p, p))) == (d, d)
        @test components(domain(@inferred TensorSpace(p, p, p))) == (d, d, d)

        @testset "spacescompatible for TensorSpace" begin
            a = PointSpace(1:4)^2
            @test ApproxFunBase.spacescompatible(a, a)
            @test ApproxFunBase.spacescompatible((a, a))
            @test ApproxFunBase.spacescompatible((a, a), (a, a))
            b = PointSpace(1:4) ⊗ PointSpace(1:3)
            @test !ApproxFunBase.spacescompatible(a, b)
            @test a == a
            @test a != b
        end
    end

    @testset "ConstantSpace" begin
        @test (@inferred convert(Fun, 2)) == Fun(2)
        f = Fun(2)
        @test (@inferred convert(Fun{typeof(space(f))}, 2)) == f

        f = Fun(2, ConstantSpace(0..1))
        g = Fun(3, ConstantSpace(0..1))
        @test f < g
        @test f <= g
        @test g > f
        @test g >= f
        @test 1 < f < 3
        @test norm(f) == 2

        @test maxspace(ConstantSpace(Point(1)), ConstantSpace(Point(2))) == ConstantSpace(Point(1) ∪ Point(2))
        @test maxspace(ConstantSpace(Point(1)), ConstantSpace(AnyDomain())) == ConstantSpace(Point(1))
        @test maxspace(ConstantSpace(AnyDomain()), ConstantSpace(Point(2))) == ConstantSpace(Point(2))
        @test maxspace(ConstantSpace(AnyDomain()), ConstantSpace(AnyDomain())) == ConstantSpace(AnyDomain())
    end

    @testset "promotion" begin
        M = Multiplication(Fun(PointSpace(1:3), [1:3;]));
        D = Derivative()
        for v in Any[[M, M], [D, D], [D, M]]
            @test (@inferred ApproxFunBase.promotedomainspace(v)) == v
            @test (@inferred ApproxFunBase.promoterangespace(v)) == v
            @test (@inferred ApproxFunBase.promotespaces(v)) == v
        end
    end

    @testset "Comparison" begin
        @test PointSpace(1:3) > ConstantSpace(0..1)
    end

    @testset "union and AmbiguousSpace" begin
        a = PointSpace(1:3)
        @test (@inferred union(a, a)) == a
        for b in Any[ApproxFunBase.UnsetSpace(), ApproxFunBase.NoSpace()]
            @test (@inferred union(a, b)) == a
            @test (@inferred union(b, a)) == a
            @test (@inferred union(b, b)) == b
        end
    end

    @testset "ArraySpace" begin
        a = ApproxFunBase.ArraySpace(PointSpace(1:4), 2)
        b = @inferred ApproxFunBase.ArraySpace(PointSpace(1:4), Val(2))
        @test a == b
        c = ApproxFunBase.ArraySpace(PointSpace(1:4), 3)
        @test a != c
        d = ApproxFunBase.ArraySpace(PointSpace(2:5), 2)
        @test a != d

        a = ApproxFunBase.ArraySpace(PointSpace(1:4), 2, 2)
        b = @inferred ApproxFunBase.ArraySpace(PointSpace(1:4), Val((2,2)))
        @test a == b

        a = @inferred space(rand(2,2))
        b = @inferred space(@SArray rand(2,2))
        @test a == b

        # empty array space
        A = ApproxFunBase.ArraySpace(empty!([PointSpace(1:3)]))
        @test length(A) == 0
        @test ApproxFunBase.dimension(A) == 0
    end
end
