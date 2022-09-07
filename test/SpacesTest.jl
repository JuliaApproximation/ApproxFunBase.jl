using ApproxFunBase, Test
import ApproxFunBase: PointSpace, HeavisideSpace, PiecewiseSegment, dimension, Vec, checkpoints
using ApproxFunOrthogonalPolynomials
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
            @test ApproxFunBase.intpow(f, 0) == f^0 == Fun(space(f), ones(ncoefficients(f)))
            for n in 1:3
                @test ApproxFunBase.intpow(f, n) == f^n == reduce(*, fill(f, n))
            end
            @test ApproxFunBase.intpow(f,-2) == f^-2 == 1/(f*f)
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

    @testset "ApproxFunOrthogonalPolynomials" begin
        @test (@inferred Fun()) == Fun(x->x, Chebyshev())
        @test (@inferred norm(Fun())) ≈ √(2/3) # √∫x^2 dx over -1..1

        v = rand(4)
        v2 = transform(NormalizedChebyshev(), v)
        @test itransform(NormalizedChebyshev(), v2) ≈ v

        f = @inferred Fun(x->x^2, Chebyshev())
        v = @inferred coefficients(f, Chebyshev(), Legendre())
        @test eltype(v) == eltype(coefficients(f))
        @test v ≈ coefficients(Fun(x->x^2, Legendre()))

        # inference check for coefficients
        v = @inferred coefficients(Float64[0,0,1], Chebyshev(), Ultraspherical(1))
        @test v ≈ [-0.5, 0, 0.5]

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

        @testset "inplace transform" begin
            @testset for sp_c in Any[Legendre(), Chebyshev(), Jacobi(1,2), Jacobi(0.3, 2.3),
                    Ultraspherical(1), Ultraspherical(2)]
                @testset for sp in Any[sp_c, NormalizedPolynomialSpace(sp_c)]
                    v = rand(10)
                    v2 = copy(v)
                    @test itransform!(sp, transform!(sp, v)) ≈ v
                    @test transform!(sp, v) ≈ transform(sp, v2)
                    @test itransform(sp, v) ≈ v2
                    @test itransform!(sp, v) ≈ v2

                    # different vector
                    p_fwd = ApproxFunBase.plan_transform!(sp, v)
                    p_inv = ApproxFunBase.plan_itransform!(sp, v)
                    @test p_inv * copy(p_fwd * copy(v)) ≈ v
                end
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

        @testset "Multivariate" begin
            @testset for S in Any[Chebyshev(), Legendre()]
                f = Fun(x->ones(2,2), S)
                @test (f+1) * f ≈ (1+f) * f ≈ f^2 + f
                @test (f-1) * f ≈ f^2 - f
                @test (1-f) * f ≈ f - f^2
                @test f + f ≈ 2f ≈ f*2
            end
        end

        @testset "static coeffs" begin
            f = Fun(Chebyshev(), SA[1,2,3])
            g = Fun(Chebyshev(), [1,2,3])
            @test coefficients(f^2) == coefficients(g^2)
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
            for k in axes(Iapprox,1), j in k + 1:min(k + bandwidths(P,2), size(Iapprox, 2))
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

        @testset "inplace ldiv" begin
            @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
                v = rand(T, 4)
                v2 = copy(v)
                ApproxFunBase.ldiv_coefficients!(Conversion(Chebyshev(), Ultraspherical(1)), v)
                @test ApproxFunBase.ldiv_coefficients(Conversion(Chebyshev(), Ultraspherical(1)), v2) ≈ v
            end
        end
    end
end
