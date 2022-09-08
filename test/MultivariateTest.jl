using ApproxFunBase: tensorizer
using Test
using ApproxFunOrthogonalPolynomials

@testset "Multivariate Tests" begin

    @testset "iterator order" begin
        S = Chebyshev()^2
        it = tensorizer(S)
        expected_order = [(1, 1)
                        (1,2)
                        (2,1)
                        (1,3)
                        (2,2)
                        (3,1)
                        (1,4)
                        (2,3)]
        k = 0
        for i in it
            k = k + 1
            if k>length(expected_order)
                break
            end
            @test i == expected_order[k]
        end
    end

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

            @test (f2_1+f2_2)(0.3, 0.5)≈f2_1(0.3, 0.5)+f2_2(0.3, 0.5)

            # 3D
            f3_1 = Fun(Chebyshev()^3, c_1)
            f3_2 = Fun(Chebyshev()^3, c_2)
            @test coefficients(f3_1+f3_2) == added_coef

            @test (f3_1+f3_2)(0.3, 0.5, 0.6)≈f3_1(0.3, 0.5, 0.6)+f3_2(0.3, 0.5, 0.6)
        end

        @testset "Multiplication" begin
            # coefficients
            c_1 = rand(20)
            c_2 = rand(30)

            # 2D
            f2_1 = Fun(Chebyshev()^2, c_1)
            f2_2 = Fun(Chebyshev()^2, c_2)

            # @test (f2_1 * f2_2)(0.4, 0.5) ≈ f2_1(0.4, 0.5) * f2_2(0.4, 0.5) broken=true

            # 3D: not implemented in code yet
            #f3_1 = Fun(Chebyshev()^3, c_1)
            #f3_2 = Fun(Chebyshev()^3, c_2)

            #@test (f3_1*f3_2)(0.4,0.5,0.6) ≈ f3_1(0.4,0.5,0.6)*f3_2(0.4,0.5,0.6)
        end
    end

end