using ApproxFunBase, Test
import ApproxFunBase: PointSpace, HeavisideSpace, PiecewiseSegment, dimension, Vec, checkpoints
using ApproxFunOrthogonalPolynomials

@testset "Spaces" begin
    @testset "PointSpace" begin
        @test eltype(domain(PointSpace([0,0.1,1])) ) == Float64

        f = @inferred Fun(x->(x-0.1),PointSpace([0,0.1,1]))
        @test roots(f) == [0.1]

        a = @inferred Fun(exp, space(f))
        @test f/a == @inferred Fun(x->(x-0.1)*exp(-x),space(f))

        f = Fun(space(f),[1.,2.,3.])

        @test (+f) == f

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

        M = Multiplication(Fun(PointSpace(1:3), [1:3;]), PointSpace(1:3))
        @test (@inferred size(M)) == (3,3)
        M2 = M + M
        @test size(M2) == (3,3)
        M = Multiplication(Fun(PointSpace(1:3), [1:3;]))
        M2 = M + M
        infty = ApproxFunBase.InfiniteCardinal{0}()
        @test (@inferred size(M2)) == (infty, infty)
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

    @testset "AmbiguousSpace" begin
        a = PointSpace(1:3)
        for b in Any[ApproxFunBase.UnsetSpace(), ApproxFunBase.NoSpace()]
            @test union(a, b) == a
            @test union(b, a) == a
            @test union(b, b) == b
        end
    end

    @testset "ApproxFunOrthogonalPolynomials" begin
        v = rand(4)
        v2 = transform(NormalizedChebyshev(), v)
        @test itransform(NormalizedChebyshev(), v2) ≈ v

        f = Fun(x->x^2, Chebyshev())
        v = coefficients(f, Chebyshev(), Legendre())
        @test v ≈ coefficients(Fun(x->x^2, Legendre()))

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
    end
end
