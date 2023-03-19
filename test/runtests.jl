using ApproxFunBase
using ApproxFunBase: ∞
using Aqua
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
using DomainSets
using FillArrays
using InfiniteArrays
using Infinities
using LinearAlgebra
using Random
using SpecialFunctions
using Test

@testset "Project quality" begin
    Aqua.test_all(ApproxFunBase, ambiguities=false)
end

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
        @test_throws MethodError ApproxFunBase.real(1,2)
        @test float(ApproxFunBase.UnsetNumber()) == ApproxFunBase.UnsetNumber()
        @test float(ApproxFunBase.UnsetNumber) == ApproxFunBase.UnsetNumber
        @test real(ApproxFunBase.UnsetNumber()) == ApproxFunBase.UnsetNumber()
        @test real(ApproxFunBase.UnsetNumber) == ApproxFunBase.UnsetNumber
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
    @testset "pad" begin
        @testset "float" begin
            @testset for T in [Float64, Any]
                a = T[1,2,3]
                b = @inferred pad(a, 4)
                @test length(b) == 4
                @test @view(b[1:3]) == a
                @test b[end] == 0
                @test pad(a, 2) == @view(a[1:2])
            end
        end
    end
    @testset "nocat" begin
        D = Derivative()
        for v in Any[
                ApproxFunBase.@nocat([D; D]),
                ApproxFunBase.@nocat(vcat(D, D))
                ]
            @test size(v) == (2,)
            @test all(==(D), v)
        end
        for v in Any[
                ApproxFunBase.@nocat([D D]),
                ApproxFunBase.@nocat(hcat(D, D))
                ]
            @test size(v) == (1,2)
            @test all(==(D), v)
        end
        v = ApproxFunBase.@nocat(hvcat((2,2), D, D, D, D))
        @test size(v) == (2,2)
        @test all(==(D), v)
    end

    @test @inferred(ApproxFunBase.flipsign(2, 0im)) == 2

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

    @testset "ChebyshevInterval" begin
        @test @inferred ApproxFunBase.domainscompatible(ChebyshevInterval{Float64}(), ChebyshevInterval{Float32}())
        @test @inferred ApproxFunBase.domainscompatible(ChebyshevInterval{Float64}(), ChebyshevInterval{BigFloat}())
    end

    @testset "union" begin
        a = ApproxFunBase.EmptyDomain()
        b = ApproxFunBase.AnyDomain()

        @test union(a, a) == a
        @test union(a, b) == a
        @test union(b, a) == a
        @test union(b, b) == b

        @testset for d in Any[a, b]
            @test union(d, 1..2) == 1..2
            @test union(1..2, d) == 1..2
        end
    end

    @testset "comparison" begin
        # Lexicographic comparison
        @test ApproxFunBase.AnyDomain() < ApproxFunBase.EmptyDomain()
        @test 1..2 < 1..3
        @test 1..2 <= 1..2
        @test 1..2 >= 1..2
        @test 1..3 >= 1..2
    end
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
            for sp2 in ((), (sp,))
                a = Multiplication(f, sp2...)
                b = Multiplication(f, sp2...)
                @test a == b
                @test bandwidths(a) == bandwidths(b)
            end
            M = Multiplication(f, sp) * Multiplication(f, sp)
            M1 = ApproxFunBase.MultiplicationWrapper(f, M)
            M2 = ApproxFunBase.MultiplicationWrapper(eltype(M), f, M)
            M3 = @inferred ApproxFunBase.MultiplicationWrapper(f, M, sp)
            @test M1 * f ≈ M2 * f ≈ M3 * f
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
                for ops in (Operator{Float64}[M, M * M], Operator{Float64}[M*M, M])
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
            @test (@inferred adjoint(T)) == adjoint(M) * adjoint(M)

            M = Multiplication(f, sp)
            @test M^2 == M * M == (T : sp)
        end
        @testset "plus operator" begin
            c = [1,2,3]
            f = Fun(PointSpace(1:3), c)
            M = Multiplication(f)
            @testset for t in [1, 3]
                op = @inferred M + t * M
                @test bandwidths(op) == bandwidths(M)
                @test coefficients(op * f) == @. (1+t)*c^2
                for op2 in (M + M + t * M, op + M)
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
            @test (@inferred Z + Z) == Z
            @test (@inferred Z + Z + Z) == Z
            @test (@inferred Z + Z + Z + Z) == Z

            @inferred (() -> (local D = Derivative(); D + D))()

            A = @inferred M + M
            @test A * f ≈ 2 * (M * f)
            M = Multiplication(f, space(f))
            A = M + M
            @test A * f ≈ 2 * (M * f)
            B = @inferred convert(Operator{ComplexF64}, A)
            @test eltype(B) == ComplexF64
            @test B * f ≈ 2 * (M * f)

            C = @inferred 2M
            D = M + C
            E = @inferred convert(Operator{ComplexF64}, D)
            @test eltype(E) == ComplexF64
            @test E * f ≈ 3 * (M * f)
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
            A = Fun(PointSpace(1:4))
            B = Multiplication(A, space(A))
            C = B * B
            @test_throws BoundsError @view C[1:2, 0:2]
            @test_throws BoundsError @view C[1:10, 1:2]

            f = Fun(PointSpace(1:10))
            M = Multiplication(f, space(f))
            T = TimesOperator([M,M])
            S = view(T, 1:2:7, 1:2:7)
            B = BandedMatrix(S)
            for I in CartesianIndices(B)
                @test B[I] ≈ S[Tuple(I)...]
            end
        end
    end
    @testset "conversion to a matrix" begin
        M = Multiplication(Fun(identity, PointSpace(1:3)))
        @test_throws ErrorException Matrix(M)
    end
    @testset "real-imag" begin
        A = (3 + 2im)*I : PointSpace(1:4)
        Ar = ApproxFunBase.real(A)
        Ai = imag(A)
        @test Ar[1:4, 1:4] == diagm(0=>fill(3, 4))
        @test Ai[1:4, 1:4] == diagm(0=>fill(2, 4))
        B = conj(A)
        Bi = imag(B)
        @test Bi[1:4, 1:4] == diagm(0=>fill(-2, 4))
        C = convert(Operator{ComplexF64}, A)
        @test C isa Operator{ComplexF64}
        @test imag(C)[1:4, 1:4] == diagm(0=>fill(2, 4))

        A = (3 - 2im)*I : PointSpace(1:4)
        Ar = ApproxFunBase.real(A)
        Ai = imag(A)
        @test Ar[1:4, 1:4] == diagm(0=>fill(3, 4))
        @test Ai[1:4, 1:4] == diagm(0=>fill(-2, 4))
    end
    @testset "tuples in promotespaces" begin
        M = Multiplication(Fun(PointSpace(1:4)), PointSpace(1:4))
        A = ApproxFunBase.promotespaces([M, M])
        B = ApproxFunBase.promotespaces((M, M))
        @test all(((x,y),) -> x == y, zip(A, B))
    end
    @testset "conversion and constantoperator" begin
        A = Conversion(PointSpace(1:4), PointSpace(1:4))
        @test convert(Number, A) == 1

        C = I : PointSpace(1:4)
        C2 = C * C
        @test domainspace(C2) == PointSpace(1:4)
        @test convert(Number, C2) == 1

        v = Float64[i^2 for i in 1:4]
        f = Fun(PointSpace(1:4), v)
        g = C * f
        @test g ≈ f
        ApproxFunBase.mul_coefficients!(Operator(2I), v)
        @test v ≈ Float64[2i^2 for i in 1:4]
    end
    @testset "Matrix types" begin
        F = Multiplication(Fun(PointSpace(1:3)), PointSpace(1:3))
        function test_matrices(F)
            A = AbstractMatrix(F)
            @test Matrix(F) == BandedMatrix(F) == BlockBandedMatrix(F) == ApproxFunBase.RaggedMatrix(F) == A
            DefBlk = ApproxFunBase.default_BlockMatrix(F)
            DefBBlk = ApproxFunBase.default_BlockBandedMatrix(F)
            DefB = ApproxFunBase.default_BandedMatrix(F)
            DefM = ApproxFunBase.default_Matrix(F)
            DefR = ApproxFunBase.default_RaggedMatrix(F)
            @test DefBlk == DefBBlk == DefB == DefM == DefR == A
        end
        test_matrices(F)
        test_matrices(im*F)
    end
end

@testset "RowVector" begin
    @testset "constructors" begin
        for s in [(2,), (1,2)], sz in [s, (s,)]
            b = fill!(ApproxFunBase.RowVector{Int}(sz...), 2)
            @test size(b) == (1,2)
            @test all(==(2), b)
        end
    end
    # for a vector of numbers, RowVector should be identical to transpose
    a = Float64.(1:4)
    at = transpose(a)
    b = ApproxFunBase.RowVector(a)
    @test b == at
    for inds in [eachindex(b), CartesianIndices(b)]
        for i in inds
            @test b[i] == at[i]
        end
    end
    M = Float64.(reshape(1:16, 4, 4))
    @test b * M == at * M
    @test b * a == at * a
    @test b * Float32.(a) == at * Float32.(a)
    @test a * b == a * at
    @test map(x->x^2, b) == map(x->x^2, at)
    @test b.^2 == at.^2
    @test hcat(b, b) == hcat(at, at)
    @test vcat(b, b) == vcat(at, at)
    @test hcat(b, 1) == hcat(at, 1)
    c = Float32[b 1]
    @test eltype(c) == Float32
    @test c == [b 1]
    c = Float32[b b]
    @test eltype(c) == Float32
    @test c == Float32[at at]

    # setindex
    b[2] = 30
    @test b[2] == b[1,2] == 30
    @test a[2] == 30

    a = rand(1)
    b = ApproxFunBase.RowVector(a)
    @test b[] == b[CartesianIndex()] == b[CartesianIndex(1)] == a[]
end

@testset "BLAS/LAPACK" begin
    @testset "gemv" begin
        # test for the pointer versions, and assert that libblastrampoline works
        @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
            a = zeros(T, 4)
            b = zeros(T, 4)
            A = Matrix{T}(I,4,4)
            x = T[1:4;]
            α, β = T(1.0), T(0.0)
            ApproxFunBase.gemv!('N', α, A, x, β, a)
            LinearAlgebra.BLAS.gemv!('N', α, A, x, β, b)
            @test a == b == x
            β = T(1.0)
            ApproxFunBase.gemv!('N', α, A, x, β, a)
            LinearAlgebra.BLAS.gemv!('N', α, A, x, β, b)
            @test a == b == 2x
        end
    end

    @testset "gemm" begin
        # test for the pointer versions, and assert that libblastrampoline works
        @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
            C1 = zeros(T, 4, 4)
            C2 = zeros(T, 4, 4)
            A = Matrix{T}(I,4,4)
            B = reshape(T[1:16;], 4, 4)
            α, β = T(1.0), T(0.0)
            ApproxFunBase.gemm!('N', 'N', α, A, B, β, C1)
            LinearAlgebra.BLAS.gemm!('N', 'N', α, A, B, β, C2)
            @test C1 == C2 == B
            β = T(1.0)
            ApproxFunBase.gemm!('N', 'N', α, A, B, β, C1)
            LinearAlgebra.BLAS.gemm!('N', 'N', α, A, B, β, C2)
            @test C1 == C2 == 2B
        end
    end

    @testset "hesseneigs" begin
        A = Float64[1 4 2 3; 1 4 1 7; 0 2 3 4; 0 0 1 3]
        λ1 = sort(ApproxFunBase.hesseneigvals(A), by = x->(real(x), imag(x)))
        λ2 = eigvals(A)
        @test λ1 ≈ λ2
        B = ComplexF64.(A)
        λ1 = sort(ApproxFunBase.hesseneigvals(B), by = x->(real(x), imag(x)))
        λ2 = eigvals(B)
        @test λ1 ≈ λ2
    end

    @testset "complexroots" begin
        cfs = [1.0, 1.0]
        r = complexroots(cfs)
        @test r == [-1.0]
        @test complexroots([0.0], cfs) == [-1.0]
    end
end

@testset "Special functions" begin
    pt = 0.5
    x = Fun(pt, ConstantSpace(1..2))
    for f in [erf, erfinv, erfc, erfcinv, erfi, gamma,
                digamma, invdigamma, trigamma, loggamma,
                airyai, airybi, airyaiprime, airybiprime,
                besselj0, besselj1, bessely0, bessely1,
                erfcx, dawson]

        @test Number(f(x)) ≈ f(pt)
        if f ∉ [erfinv, erfcinv, invdigamma]
            @test Number(f(x*im)) ≈ f(pt*im)
        end
    end
    for f in [besselj, bessely, besseli, besselk, besselkx,
              hankelh1, hankelh2, hankelh1x, hankelh2x]
        @test Number(f(1, x)) ≈ f(1, pt)
    end
    @test Number(logabsgamma(x)[1]) ≈ logabsgamma(pt)[1]
    @test logabsgamma(x)[2] ≈ logabsgamma(pt)[2]
    @test Number(gamma(2, x)) ≈ gamma(2, pt)
    @test Number(gamma(2, x*im)) ≈ gamma(2, pt*im)
end

@testset "ToeplitzOperator" begin
    A = @inferred ApproxFunBase.SymToeplitzOperator(Int[])
    B = A[1:5, 1:5]
    @test all(iszero, B)

    @testset "kronecker type stability" begin
        T = ApproxFunBase.ToeplitzOperator(Float64[1,2], Float64[1,2])
        TT = T ⊗ T
        TypeExp = Union{ApproxFunBase.SubOperator{Float64, KroneckerOperator{ToeplitzOperator{Float64}, ToeplitzOperator{Float64}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, Tuple{Int64, Int64}, Tuple{Infinities.InfiniteCardinal{0}, Infinities.InfiniteCardinal{0}}}, ApproxFunBase.SubOperator{Float64, KroneckerOperator{ToeplitzOperator{Float64}, ToeplitzOperator{Float64}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, Tuple{Int64, Int64}, Tuple{Int64, Int64}}}
        @inferred TypeExp view(T, 1:1, 1:1)
        TypeExp2 = Union{ApproxFunBase.SubOperator{Float64, KroneckerOperator{ToeplitzOperator{Float64}, ToeplitzOperator{Float64}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, Float64}, Tuple{StepRange{Int64, Int64}, StepRange{Int64, Int64}}, Tuple{Int64, Int64}, Tuple{Infinities.InfiniteCardinal{0}, Infinities.InfiniteCardinal{0}}}, ApproxFunBase.SubOperator{Float64, KroneckerOperator{ToeplitzOperator{Float64}, ToeplitzOperator{Float64}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, FillArrays.Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, Float64}, Tuple{StepRange{Int64, Int64}, StepRange{Int64, Int64}}, Tuple{Int64, Int64}, Tuple{Int64, Int64}}}
        @inferred TypeExp2 view(T, 1:1:1, 1:1:1)
        TypeExp3 = Union{ApproxFunBase.SubOperator{Float64, KroneckerOperator{ToeplitzOperator{Float64}, ToeplitzOperator{Float64}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, Float64}, Tuple{InfiniteArrays.InfUnitRange{Int64}, InfiniteArrays.InfUnitRange{Int64}}, Tuple{Infinities.Infinity, Infinities.Infinity}, Tuple{Infinities.InfiniteCardinal{0}, Infinities.InfiniteCardinal{0}}}, ApproxFunBase.SubOperator{Float64, KroneckerOperator{ToeplitzOperator{Float64}, ToeplitzOperator{Float64}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, TensorSpace{Tuple{SequenceSpace, SequenceSpace}, DomainSets.VcatDomain{2, Int64, (1, 1), Tuple{ApproxFunBase.PositiveIntegers, ApproxFunBase.PositiveIntegers}}, Union{}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, ApproxFunBase.CachedIterator{Tuple{Int64, Int64}, ApproxFunBase.Tensorizer{Tuple{Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}, Ones{Int64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}}}, Float64}, Tuple{InfiniteArrays.InfUnitRange{Int64}, InfiniteArrays.InfUnitRange{Int64}}, Tuple{Infinities.Infinity, Infinities.Infinity}, Tuple{Int64, Int64}}}
        @inferred TypeExp3 view(T, 1:∞, 1:∞)
    end
end

@testset "Tensorizer" begin
    @testset "TrivialTensorizer" begin
        @testset "2D" begin
            ax = Ones{Int}(∞)
            t = ApproxFunBase.Tensorizer((ax,ax))
            @test (@inferred length(t)) == length(ax)^2
            v = collect(Iterators.take(t, 150))
            @test eltype(v) == eltype(t)
            @testset for (i, vi) in enumerate(v)
                blk = ApproxFunBase.block(t, i)
                @test i ∈ ApproxFunBase.blockrange(t, blk)
                @test vi == t[i]
                @test findfirst(vi, t) == i
            end
        end
        @testset "nD" begin
            ax = Ones{Int}(∞)
            t = ApproxFunBase.Tensorizer((ax,ax,ax))
            v = collect(Iterators.take(t, 150))
            @test eltype(v) == eltype(t)
            @testset for (i,vi) in enumerate(v)
                blk = ApproxFunBase.block(t, i)
                @test i ∈ ApproxFunBase.blockrange(t, blk)
            end
        end
    end
    @testset "Tensorizer2D" begin
        ax = Ones{Int}(4)
        t = ApproxFunBase.Tensorizer((ax,ax))
        v = collect(Iterators.take(t, 16))
        @test eltype(v) == eltype(t)
        @testset for (i, vi) in enumerate(v)
            @test findfirst(vi, t) == i
        end
    end
    @testset "cache" begin
        ax = Ones{Int}(4);
        t = ApproxFunBase.Tensorizer((ax,ax))
        c = ApproxFunBase.cache(t)
        v = collect(c)
        @testset for i in eachindex(c)
            @test c[i] == v[i]
        end
    end
end

@testset "CachedIterator" begin
    v = ApproxFunBase.CachedIterator(1:4)
    @test Base.IteratorSize(v) == Base.HasLength()
    @test length(v) == 4
    @test eltype(v) == Int
    @test findfirst(3, v) == 3
    vs = collect(v)
    @test vs == [1:4;]
    @test eltype(vs) == Int

    v = ApproxFunBase.CachedIterator(Iterators.take(1:14, 4))
    @test Base.IteratorSize(v) == Base.HasLength()
    @test length(v) == 4
    @test eltype(v) == Int
    @test findfirst(3, v) == 3
    vs = collect(v)
    @test vs == [1:4;]
    @test eltype(vs) == Int

    v = ApproxFunBase.CachedIterator(1:∞)
    @test Base.IteratorSize(v) == Base.IsInfinite()
    @test isinf(length(v))
    @test eltype(v) == Int
    @test findfirst(3, v) == 3

    s = PointSpace(1:4) + PointSpace(1:4)
    b = ApproxFunBase.BlockInterlacer(s);
    it = ApproxFunBase.CachedIterator(b)
    # Test that the iterator isn't stateful
    @test collect(it) == collect(Iterators.take(it, length(it)))
    @test collect(it) == collect(b)
end

@time include("ETDRK4Test.jl")
include("show.jl")
