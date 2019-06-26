using ApproxFun

@testset "MatrixOperator" begin
    A=[Legendre(),Chebyshev()]
    B=[Chebyshev(),Legendre()]
    plan=[2,1]
    C=Conversion(A,B,plan)
    @test C[1,1] == ZeroOperator(Legendre(),Chebyshev())
    @test C[1,2] == Conversion(Chebyshev(),Chebyshev())
    @test C[2,1] == Conversion(Legendre(),Legendre())
    @test C[2,2] == ZeroOperator(Chebyshev(),Legendre())
    
    A=[JacobiWeight(1,0,Jacobi(1,-1)),JacobiWeight(2/3,0,Jacobi(2/3,-2/3)),JacobiWeight(1/3,0,Jacobi(1/3,-1/3))]
    B=[JacobiWeight(2/3,0,Jacobi(2/3,-2/3)),JacobiWeight(1/3,0,Jacobi(1/3,-1/3)),Legendre()]
    plan=[3,1,2]
    C=Conversion(A,B,plan)
    @test C[1,1] == ZeroOperator(JacobiWeight(1,0,Jacobi(1,-1)),JacobiWeight(2/3,0,Jacobi(2/3,-2/3)))
    @test C[1,2] == Conversion(JacobiWeight(2/3,0,Jacobi(2/3,-2/3)),JacobiWeight(2/3,0,Jacobi(2/3,-2/3)))
    @test C[2,3] == Conversion(JacobiWeight(1/3,0,Jacobi(1/3,-1/3)),JacobiWeight(1/3,0,Jacobi(1/3,-1/3)))
    @test C[3,1] == Conversion(JacobiWeight(1,0,Jacobi(1,-1)),Legendre())
end
