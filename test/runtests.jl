using ApproxFunBase, LinearAlgebra, Test
    import ApproxFunBase: Infinity, ∞

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

    # TODO: Tensorizer tests
end

@testset "Domain" begin
    @test 0.45-0.65im ∉ Segment(-1,1)

    @test ApproxFunBase.AnySegment() == ApproxFunBase.AnySegment()

    @test ApproxFunBase.dimension(Domain{Float64}) == 1
    @test ApproxFunBase.dimension(Segment{Float64}) == 1
    @test ApproxFunBase.dimension(ChebyshevInterval()) == 1
    @test ApproxFunBase.dimension(ChebyshevInterval()^2) == 2
    @test ApproxFunBase.dimension(ChebyshevInterval()^3) == 3

    @test isambiguous(convert(ApproxFunBase.Point,ApproxFunBase.AnyDomain()))
    @test isambiguous(ApproxFunBase.Point(ApproxFunBase.AnyDomain()))

    @test_skip ApproxFunBase.Point(NaN) == ApproxFunBase.Point(NaN)
end

@time include("MatrixTest.jl")
@time include("SpacesTest.jl")
