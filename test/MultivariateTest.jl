using ApproxFunBase, Test
using ApproxFunOrthogonalPolynomials

@testset "Multivariate Tests" begin
    @testset "Evaluation" begin
        
        # 2D
        f2 = Fun(Chebyshev()^2, [1.0])
        @test f2(0.2, 0.4) == 1.0

        # 3D
        f3 = Fun(Chebyshev()^3, [1.0])
        @test f3(0.2, 0.4, 0.2) == 1.0
    end

    @testset "Arithmetic" begin
        @testset "Addition" begin
           # coefficients
            c_1 = rand(20)
            c_2 = rand(30)

            added_coef = [c_2[1:20]+c_1;c_2[21:end]]

            # 2D
            f2_1 = Fun(Chebyshev()^2, c_1)
            f2_2 = Fun(Chebyshev()^2, c_2)
            @test coefficients(f2_1+f2_2) == added_coef

            # 3D
            f3_1 = Fun(Chebyshev()^3, c_1)
            f3_2 = Fun(Chebyshev()^3, c_2)
            @test coefficients(f2_1+f2_2) == added_coef 
        end

        @testset "Multiplication" begin
            # coefficients
            c_1 = rand(20)
            c_2 = rand(30)

            # 2D
            f2_1 = Fun(Chebyshev()^2, c_1)
            f2_2 = Fun(Chebyshev()^2, c_2)

            @test (f2_1 * f2_2)(0.4, 0.5) ≈ f2_1(0.4, 0.5) * f2_2(0.4, 0.5) broken=true

            # 3D: not implemented in code yet
            #f3_1 = Fun(Chebyshev()^3, c_1)
            #f3_2 = Fun(Chebyshev()^3, c_2)

            #@test (f3_1*f3_2)(0.4,0.5,0.6) ≈ f3_1(0.4,0.5,0.6)*f3_2(0.4,0.5,0.6)
        end
    end

end