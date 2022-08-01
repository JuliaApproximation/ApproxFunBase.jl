using ApproxFunBase, LinearAlgebra, Random, Test
import ApproxFunBase: ∞

@testset "Helper" begin
    @testset "interlace" begin
        @test ApproxFunBase.interlace!([-1.0],0) == [-1.0]
        @test ApproxFunBase.interlace!([1.0,2.0],0) == [2.0,1.0]
        @test ApproxFunBase.interlace!([1,2,3],0) == [2,1,3]
        @test ApproxFunBase.interlace!([1,2,3,4],0) == [3,1,4,2]

        @test ApproxFunBase.interlace!([-1.0],1) == [-1.0]
        @test ApproxFunBase.interlace!([1.0,2.0],1) == [1.0,2.0]
        @test ApproxFunBase.interlace!([1,2,3],1) == [1,3,2]
        @test ApproxFunBase.interlace!([1,2,3,4],1) == [1,3,2,4]

        @test ApproxFunBase.interlace(collect(6:10),collect(1:5)) == ApproxFunBase.interlace!(collect(1:10),0)
        @test ApproxFunBase.interlace(collect(1:5),collect(6:10)) == ApproxFunBase.interlace!(collect(1:10),1)
    end

    @testset "Iterators" begin
        @test cache(ApproxFunBase.BlockInterlacer((1:∞,[2],[2])))[1:6] ==
            [(1,1),(2,1),(2,2),(3,1),(3,2),(1,2)]

        @test collect(ApproxFunBase.BlockInterlacer(([2],[2],[2]))) ==
            [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]
    end

    @testset "issue #94" begin
        @test ApproxFunBase.real !== Base.real
        @test_throws MethodError ApproxFunBase.real(1,2)
    end

    @testset "hasnumargs" begin
        onearg(x) = x
        twoargs(x, y) = x + y
        @test ApproxFunBase.hasnumargs(onearg, 1)
        @test ApproxFunBase.hasnumargs(twoargs, 2)
    end
    @testset "don't pirate dot" begin
        @test ApproxFunBase.dot !== LinearAlgebra.dot
        struct DotTester end
        # check that unknown types don't lead to a stack overflow
        @test_throws MethodError ApproxFunBase.dot(DotTester())
    end

    # TODO: Tensorizer tests
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

    @test Segment(-1,1) .+ 1 ≡ Segment(0,2)
    @test 2 .* Segment(-1,1) .+ 1 ≡ Segment(-1,3)
    @test Segment(-1,1) .^ 2 ≡ Segment(0,1)
    @test Segment(1,-1) .^ 2 ≡ Segment(1,0)
    @test Segment(1,2) .^ 2 ≡ Segment(1,4)
    @test sqrt.(Segment(1,2)) ≡ Segment(1,sqrt(2))
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

@testset "empty coefficients" begin
    v = Float64[]
    f = Fun(PointSpace(), v)
    # empty coefficients should short-circuit
    @test ApproxFunBase.coefficients(f) === v
end

@testset "operator" begin
    @testset "operator algebra" begin
        @testset "Multiplication" begin
            sp = PointSpace(1:3)
            coeff = [1:3;]
            f = Fun(sp, coeff)
            for sp2 in Any[(), (sp,)]
                a = Multiplication(f, sp2...)
                b = Multiplication(f, sp2...)
                @test a == b
                @test bandwidths(a) == bandwidths(b)
            end
        end
        @testset "TimesOperator" begin
            sp = PointSpace(1:3)
            coeff = [1:3;]
            f = Fun(sp, coeff)
            for sp2 in Any[(), (sp,)]
                M = Multiplication(f, sp2...)
                a = (M * M) * M
                b = M * (M * M)
                @test a == b
                @test bandwidths(a) == bandwidths(b)
            end
            @testset "unwrap TimesOperator" begin
                M = Multiplication(f)
                for ops in Any[Operator{Float64}[M, M * M], Operator{Float64}[M*M, M]]
                    @test TimesOperator(ops).ops == [M, M, M]
                end
            end
            M = Multiplication(f)
            @test coefficients(((M * M) * M) * f) == coefficients((M * M * M) * f)
            T = @inferred TimesOperator(M, M)
            TM = @inferred TimesOperator(T, M)
            MT = @inferred TimesOperator(M, T)
            TT = @inferred TimesOperator(T, T)
            @test T == M * M
            @test TM == T * M
            @test MT == M * T
            @test T * M == M * T == M * M * M
            @test TT == T * T == M * M * M * M
        end
        @testset "plus operator" begin
            c = [1,2,3]
            f = Fun(PointSpace(1:3), c)
            M = Multiplication(f)
            @testset for t in [1, 3]
                op = M + t * M
                @test bandwidths(op) == bandwidths(M)
                @test coefficients(op * f) == @. (1+t)*c^2
                for op2 in Any[M + M + t * M, op + M]
                    @test bandwidths(op2) == bandwidths(M)
                    @test coefficients(op2 * f) == @. (2+t)*c^2
                end
                op3 = op + op
                @test bandwidths(op3) == bandwidths(M)
                @test coefficients(op3 * f) == @. 2(1+t)*c^2

                f1 = (op + op - op)*f
                f2 = ((op + op) - op)*f
                f3 = op * f
                @test coefficients(f1) == coefficients(f2) == coefficients(f3)
            end
            Z = ApproxFunBase.ZeroOperator()
            @test Z + Z == Z
            @test Z + Z + Z == Z
            @test Z + Z + Z + Z == Z
        end
    end

    @testset "operator indexing" begin
        @testset "SubOperator" begin
            D = Dirichlet(ConstantSpace(0..1))
            S = D[:, :]
            @test S[1,1] == 1
            ax1 = axes(S, 1)
            ax2 = axes(S, 2)
            inds1 = Any[ax1, StepRange(ax1), :]
            inds2 = Any[ax2, StepRange(ax2), :]
            @testset for r2 in inds2, r1 in inds1
                M = S[r1, r2]
                @test M isa AbstractMatrix
                @test size(M) == (2,1)
                @test all(==(1), M)
            end
            @testset for r1 in inds1
                V = S[r1, 1]
                @test V isa AbstractVector
                @test size(V) == (2,)
                @test all(==(1), V)
            end
            @testset for r2 in inds2
                V = S[1, r2]
                @test V isa AbstractVector
                @test size(V) == (1,)
                @test all(==(1), V)
            end
        end
    end

    @testset "conversion to a matrix" begin
        M = Multiplication(Fun(identity, PointSpace(1:3)))
        @test_throws ErrorException Matrix(M)
    end
end

@time include("ETDRK4Test.jl")
