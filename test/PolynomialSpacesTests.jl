module MiscApproxFunBaseTests

using ApproxFunBase
using Test
using LinearAlgebra
using StaticArrays
using BandedMatrices

using IntervalSets: AbstractInterval
import IntervalSets: endpoints, closedendpoints

using SpecialFunctions

include("AFOP/PolynomialSpaces.jl")
using .PolynomialSpaces

struct UniqueInterval{T, I<:AbstractInterval{T}} <: AbstractInterval{T}
    parentinterval :: I
end

for f in [:endpoints, :closedendpoints]
    @eval $f(m::UniqueInterval) = $f(m.parentinterval)
end

Base.in(x, m::UniqueInterval) = in(x, m.parentinterval)
Base.isempty(m::UniqueInterval) = isempty(m.parentinterval)

ApproxFunBase.domainscompatible(a::UniqueInterval, b::UniqueInterval) = a == b

Base.:(==)(a::UniqueInterval, b::UniqueInterval) = (@assert a.parentinterval == b.parentinterval; true)

@testset "PolynomialSpaces" begin
    @testset "Constructor" begin
        @test (@inferred Fun()) == Fun(x->x, Chebyshev())
        @test (@inferred norm(Fun())) ≈ norm(Fun(), 2) ≈ √(2/3) # √∫x^2 dx over -1..1
    end

    @testset "transform" begin
        v = rand(4)
        v2 = transform(NormalizedChebyshev(), v)
        @test itransform(NormalizedChebyshev(), v2) ≈ v

        @testset "coefficients" begin
            f = @inferred Fun(x->x^2, Chebyshev())
            v = @inferred coefficients(f, Chebyshev(), Legendre())
            @test eltype(v) == eltype(coefficients(f))
            @test v ≈ coefficients(Fun(x->x^2, Legendre()))

            # inference check for coefficients
            v = @inferred coefficients(Float64[0,0,1], Chebyshev(), Ultraspherical(1))
            @test v ≈ [-0.5, 0, 0.5]
        end
    end

    @testset "multiplication inference" begin
        function g2()
           f = Fun(0..1)
           f * f
        end
        y = @inferred g2()(0.1)
        @test y ≈ (0.1)^2

        function g3()
           f = Fun(0..1)
           f * f * f
        end
        y = @inferred g3()(0.1)
        @test y  ≈ (0.1)^3

        function g4()
           f = Fun(0..1)
           f * f * f * f
        end
        y = @inferred g4()(0.1)
        @test y ≈ (0.1)^4
    end

    @testset "intpow" begin
        @testset "Interval" begin
            function h(::Val{N}) where {N}
               f = Fun(0..1)
               f^N
            end
            @test (@inferred h(Val(0)))(0.1) ≈ (0.1)^0
            @test (@inferred h(Val(1)))(0.1) ≈ (0.1)^1
            @test (@inferred h(Val(2)))(0.1) ≈ (0.1)^2
            @test (@inferred h(Val(3)))(0.1) ≈ (0.1)^3
            @test (@inferred h(Val(4)))(0.1) ≈ (0.1)^4
            @test h(Val(10))(0.1) ≈ (0.1)^10 rtol=1e-6
        end

        @testset "ChebyshevInterval" begin
            function h(::Val{N}) where {N}
               f = Fun()
               f^N
            end
            @test (@inferred h(Val(0)))(0.1) ≈ (0.1)^0
            @test (@inferred h(Val(1)))(0.1) ≈ (0.1)^1
            @test (@inferred h(Val(2)))(0.1) ≈ (0.1)^2
            @test (@inferred h(Val(3)))(0.1) ≈ (0.1)^3
            @test (@inferred h(Val(4)))(0.1) ≈ (0.1)^4
            @test h(Val(10))(0.1) ≈ (0.1)^10 rtol=1e-6
        end

        @testset "UniqueInterval" begin
            f = Fun(UniqueInterval(0..1))
            g = @inferred f^0
            @test coefficients(g) == coefficients(one(Fun(0..1)))
            g = @inferred f^1
            @test coefficients(g) == coefficients(Fun(0..1)^1)
            g = @inferred f^2
            @test coefficients(g) == coefficients(Fun(0..1)^2)
            g = @inferred f*f
            @test coefficients(g) == coefficients(Fun(0..1)^2)
            g = @inferred f^3
            @test coefficients(g) == coefficients(Fun(0..1)^3)
            g = @inferred f*f*f
            @test coefficients(g) == coefficients(Fun(0..1)^3)
            g = @inferred f^4
            @test coefficients(g) == coefficients(Fun(0..1)^4)
            g = @inferred f*f*f*f
            @test coefficients(g) == coefficients(Fun(0..1)^4)
        end
    end

    @testset "int coeffs" begin
        f = Fun(Chebyshev(), [0,1])
        @test f(0.4) ≈ 0.4
        f = Fun(NormalizedChebyshev(), [0,1])
        @test f(0.4) ≈ 0.4 * √(2/pi)

        f = Fun(Chebyshev(), [1])
        @test f(0.4) ≈ 1
        f = Fun(NormalizedChebyshev(), [1])
        @test f(0.4) ≈ √(1/pi)
    end

    @testset "pad" begin
        @testset "Fun" begin
            f = Fun()
            zf = zero(f)
            @test (@inferred pad([f], 3)) == [f, zf, zf]
            @test (@inferred pad([f, zf], 1)) == [f]
            v = [f, zf]
            @test @inferred pad!(v, 1) == [f]
            @test length(v) == 1
        end
    end

    @testset "conversion" begin
        C12 = Conversion(Chebyshev(), NormalizedLegendre())
        C21 = Conversion(NormalizedLegendre(), Chebyshev())
        @test Matrix((C12 * C21)[1:10, 1:10]) ≈ I
        @test Matrix((C21 * C12)[1:10, 1:10]) ≈ I

        C12 = Conversion(Chebyshev(), NormalizedPolynomialSpace(Ultraspherical(1)))
        C1C2 = Conversion(Ultraspherical(1), NormalizedPolynomialSpace(Ultraspherical(1))) *
                Conversion(Chebyshev(), Ultraspherical(1))
        @test Matrix(C12[1:10, 1:10]) ≈ Matrix(C1C2[1:10, 1:10])
    end

    @testset "union" begin
        @test union(Chebyshev(), NormalizedLegendre()) == Jacobi(Chebyshev())
        @test union(Chebyshev(), Legendre()) == Jacobi(Chebyshev())
    end

    @testset "Fun constructor" begin
        # we make the fun go through somewhat complicated chains of functions
        # that break inference of the space
        # however, the type of coefficients should be inferred correctly.
        f = Fun(Chebyshev(0..1))
        newfc(f) = coefficients(Fun(Fun(f, Legendre(0..1)), space(f)))
        newvals(f) = values(Fun(Fun(f, Legendre(0..1)), space(f)))
        @test newfc(f) ≈ coefficients(f)
        @test newvals(f) ≈ values(f)

        newfc2(f) = coefficients(chop(pad(f, 10)))
        @test newfc2(f) == coefficients(f)

        f2 = Fun(space(f), view(Float64[1:4;], :))
        f3 = Fun(space(f), Float64[1:4;])
        @test newvals(f2) ≈ values(f3)
        @test values(f2) ≈ values(f3)

        # Ensure no trailing zeros
        f = Fun(Ultraspherical(0.5, 0..1))
        cf = coefficients(f)
        @test findlast(!iszero, cf) == length(cf)

        @testset "OneHotVector" begin
            for n in [1, 3, 10_000]
                f = Fun(Chebyshev(), [zeros(n-1); 1])
                g = ApproxFunBase.basisfunction(Chebyshev(), n)
                @test f == g
                @test f(0.5) == g(0.5)
            end
        end
    end

    @testset "multiplication of Funs" begin
        f = Fun(Chebyshev(), Float64[1:101;])
        g = Fun(Chebyshev(), Float64[1:101;]*im)
        @test f(0.5)*g(0.5) ≈ (f*g)(0.5)
    end

    @testset "ArraySpace" begin
        @testset for S in Any[Chebyshev(), Legendre()]
            f = Fun(x->ones(2,2), S)
            @test (f+1) * f ≈ (1+f) * f ≈ f^2 + f
            @test (f-1) * f ≈ f^2 - f
            @test (1-f) * f ≈ f - f^2
            @test f + f ≈ 2f ≈ f*2
        end
    end

    @testset "Multivariate" begin
        @testset "kron" begin
            x = Fun()
            O = x ⊗ I
            @test O * Fun((x,y)->x^2 * y^2, Chebyshev()^2) ≈ Fun((x,y)->x^3 * y^2, Chebyshev()^2)
            O = I ⊗ x
            @test O * Fun((x,y)->x^2 * y^2, Chebyshev()^2) ≈ Fun((x,y)->x^2 * y^3, Chebyshev()^2)
        end
    end

    @testset "static coeffs" begin
        f = Fun(Chebyshev(), SA[1,2,3])
        g = Fun(Chebyshev(), [1,2,3])
        @test coefficients(f^2) == coefficients(g^2)
    end

    @testset "special functions" begin
        for f in Any[Fun(), Fun(-0.5..1), Fun(Segment(1.0+im,2.0+2im))]
            for spfn in Any[sin, cos, exp]
                p = leftendpoint(domain(f))
                @test spfn(f)(p) ≈ spfn(p) atol=1e-14
            end
        end
    end

    @testset "Derivative" begin
        @test Derivative() == Derivative()
        for d in Any[(), (0..1,)]
            for ST in Any[Chebyshev, Legendre,
                    (x...) -> Jacobi(2,2,x...), (x...) -> Jacobi(1.5,2.5,x...)]
                S1 = ST(d...)
                for S in [S1, NormalizedPolynomialSpace(S1)]
                    @test Derivative(S) == Derivative(S,1)
                    @test Derivative(S)^2 == Derivative(S,2)
                    f = Fun(x->x^3, S)
                    @test Derivative(S) * f ≈ Fun(x->3x^2, S)
                    @test Derivative(S,2) * f ≈ Fun(x->6x, S)
                    @test Derivative(S,3) * f ≈ Fun(x->6, S)
                    @test Derivative(S,4) * f ≈ zeros(S)
                end
            end
        end
        @test Derivative(Chebyshev()) != Derivative(Chebyshev(), 2)
        @test Derivative(Chebyshev()) != Derivative(Legendre())
    end

    @testset "SubOperator" begin
        D = Derivative(Chebyshev())
        S = @view D[1:10, 1:10]
        @test rowrange(S, 1) == 2:2
        @test colrange(S, 2) == 1:1
        @test (@inferred BandedMatrix(S)) == (@inferred Matrix(S))

        A = Derivative() * Multiplication(Fun()) : Chebyshev();
        kr = 1:ApproxFunBase.InfiniteCardinal{0}()
        B1 = A[kr, :][1:10, 1:10]
        B2 = A[:, kr][1:10, 1:10]
        B3 = A[:, :][1:10, 1:10]
        B4 = A[kr, kr][1:10, 1:10]
        @test B1 == B2 == B3 == B4
    end

    @testset "CachedOperator" begin
        C = cache(Derivative())
        C = C : Chebyshev() → Ultraspherical(2)
        D = Derivative() : Chebyshev() → Ultraspherical(2)
        @test C[1:2, 1:0] == D[1:2, 1:0]
        @test C[1:10, 1:10] == D[1:10, 1:10]
        for col in 1:5, row in 1:5
            @test C[row, col] == D[row, col]
        end
    end

    @testset "PartialInverseOperator" begin
        @testset "sanity check" begin
            A = UpperTriangular(rand(10, 10))
            B = inv(A)
            for I in CartesianIndices(B)
                @test B[I] ≈ ApproxFunBase._getindexinv(A, Tuple(I)..., UpperTriangular)
            end
        end
        C = Conversion(Chebyshev(), Ultraspherical(1))
        P = PartialInverseOperator(C, (0, 6))
        Iapprox = (P * C)[1:10, 1:10]
        @test all(isone, diag(Iapprox))
        for k in axes(Iapprox,1), j in k + 1:min(k + bandwidth(P,2), size(Iapprox, 2))
            @test Iapprox[k,j] ≈ 0 atol=eps(eltype(Iapprox))
        end
        B = AbstractMatrix(P[1:10, 1:10])
        @testset for I in CartesianIndices(B)
            @test B[I] ≈ P[Tuple(I)...] rtol=1e-8 atol=eps(eltype(B))
        end
    end

    @testset "istriu/istril" begin
        for D in Any[Derivative(Chebyshev()),
                Conversion(Chebyshev(), Legendre()),
                Multiplication(Fun(Chebyshev()), Chebyshev())]
            D2 = D[1:3, 1:3]
            for f in Any[istriu, istril]
                @test f(D) == f(D2)
                @test f(D') == f(D2')
            end
        end
    end

    @testset "layout Operators" begin
        D = Derivative(Chebyshev())
        for S in (SymmetricOperator(D),
                    HermitianOperator(D))
            S2 = convert(Operator{Int}, S)
            B = S2[1:10, 1:10]
            @test B isa BandedMatrix{Int}
            @test S2[1:10, 1:10] == S[1:10, 1:10]
        end
    end

    @testset "inplace ldiv" begin
        @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
            v = rand(T, 4)
            v2 = copy(v)
            ApproxFunBase.ldiv_coefficients!(Conversion(Chebyshev(), Ultraspherical(1)), v)
            @test ApproxFunBase.ldiv_coefficients(Conversion(Chebyshev(), Ultraspherical(1)), v2) ≈ v
        end
    end

    @testset "specialfunctionnormalizationpoint" begin
        a = @inferred ApproxFunBase.specialfunctionnormalizationpoint(exp,real,Fun())
        @test a[1] == 1
        @test a[2] ≈ exp(1)
    end

    @testset "ArraySpace" begin
        f = Fun(x->[cos(x), exp(x)], -1..1)
        for x in [-1,0,1]
            @test (f + one(f))(x) ≈ f(x) .+ one.(f(x))
        end
    end

    @testset "Bivariate" begin
        f = (x,y)->x^2 * y^3
        P = ProductFun(f, Chebyshev()⊗Chebyshev())
        x = 0.2; y = 0.3;
        @test P(x, y) ≈ f(x,y)

        @test (P * one(P))(x,y) ≈ P(x,y)
        @test (P + zero(P))(x,y) ≈ P(x,y)

        # coefficients copied from P above
        coeffs = SVector{4}(zeros(3), [0.375, 0, 0.375], zeros(3), [0.125, 0, 0.125])
        fv = [Fun(Chebyshev(), c) for c in coeffs]
        P2 = @inferred ProductFun(fv, Chebyshev())
        @test P2(x, y) ≈ f(x, y)

        L = LowRankFun(f, Chebyshev()⊗Chebyshev())
        @test (@inferred L * P)(x, y) ≈ (@inferred P * L)(x, y) ≈ f(x,y)^2

        @test (L^0 * L)(x,y) ≈ L(x,y)
        @test (@inferred((L -> L^1)(L)))(x,y) ≈ L(x,y)
        @test (@inferred((L -> L^2)(L)))(x,y) ≈ (L*L)(x,y)
        @test (@inferred((L -> L^4)(L)))(x,y) ≈ (L*L*L*L)(x,y)
        @test (L * one(L))(x,y) ≈ L(x,y)
        @test (L + zero(L))(x,y) ≈ L(x,y)
    end

    @testset "Multiplication" begin
        # empty coefficients
        f = () -> Multiplication(Fun(Chebyshev(), Float64[]), Ultraspherical(1))
        M = VERSION >= v"1.8" ? (@inferred f()) : f()
        @test all(iszero, coefficients(M * Fun()))
    end

    @testset "TimesOperator" begin
        x = Fun()
        A = @inferred Derivative() * Multiplication(x, Chebyshev())
        @test A * x ≈ 2x
    end

    @testset "ConstantSpace" begin
        S = Chebyshev()
        d = domain(S)
        C = ConstantSpace(d)
        @test promote_type(typeof(S), typeof(C)) == typeof(S)
        @test promote_type(typeof(S|(2:3)), typeof(C)) <: Space{typeof(d)}

        @test union(S, C) == S
        # space doesn't contain constant
        cmps = union(S|(2:4), C)
        @test C in components(cmps)
        @test S|(2:4) in components(cmps)

        @test promote_type(typeof(Fun()), Float64) == typeof(Fun())
        @test [Fun(), 1] isa Vector{typeof(Fun())}
        @test promote_type(Fun{typeof(Chebyshev())}, Float64) == typeof(Fun())
    end
end

@testset "Special Functions" begin
    g = Fun(x ->x^2, 0.5..1)
    @testset for f in [
            sinpi, cospi, sin, cos, cosh, exp2, exp10, log2, log10, csc, sec,
              cot, acot, sinh, csch, asinh, acsch,
              sech, tanh, coth,
              sinc, cosc, log1p, log, expm1, tan,
              cbrt, sqrt, abs, abs2, sign, inv,
              angle]

        @test f(g)(0.75) ≈ f((0.75)^2)
    end
    @test abs2(Fun())(-0.4) ≈ 0.4^2
    @test sign(Fun())(-0.4) ≈ -1

    @test argmax(Fun()) == 1
    @test argmin(Fun()) == -1
    @test maximum(Fun()) == 1
    @test minimum(Fun()) == -1
    @test extrema(Fun()) == (-1,1)

    @testset "SpecialFunctions" begin
        g = Fun(0.1..0.4)
        for f in [erfcx, dawson, erf, erfi,
                    airyai, airybi, airyaiprime, airybiprime,
                    erfcinv, eta, gamma, invdigamma, digamma,
                    trigamma,
                ]
            @test f(g)(0.3) ≈ f(0.3)
        end
    end
end

end # module
