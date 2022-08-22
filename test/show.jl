@testset "show" begin
	@testset "Domain" begin
		@testset "Segment" begin
			s = Segment(0, 1)
			@test contains(repr(s), repr(leftendpoint(s)))
			@test contains(repr(s), repr(rightendpoint(s)))
		end
	end
	@testset "Space" begin
		@testset "ConstantSpace" begin
			@test contains(repr(ConstantSpace()), "ConstantSpace")
			c = ConstantSpace(0..1)
			@test contains(repr(c), "ConstantSpace")
			@test contains(repr(c), repr(domain(c)))
		end
	end
	@testset "Fun" begin
		f = Fun(PointSpace(1:3), [1,2,3])
		s = repr(f)
		@test startswith(s, "Fun")
		@test contains(s, repr(space(f)))
		@test contains(s, repr(coefficients(f)))

		f = Fun(ConstantSpace(0..1), [2])
		@test contains(repr(f), repr(coefficient(f, 1)))
		@test contains(repr(f), repr(domain(f)))

		f = Fun(ConstantSpace(), [2])
		@test contains(repr(f), repr(coefficient(f, 1)))
	end
	@testset "Operator" begin
		@testset "Derivative" begin
			D = Derivative()
			dsum = ApproxFunBase.summarystr(D)
			@test repr(D) == dsum
			io = IOBuffer()
			show(io, MIME"text/plain"(), D)
			@test contains(String(take!(io)), dsum)

			D = Derivative(Chebyshev())
			dsum = ApproxFunBase.summarystr(D)
			show(io, MIME"text/plain"(), D)
			@test contains(String(take!(io)), dsum)
		end
		@testset "SubOperator" begin
			D = Derivative(Chebyshev())
			S = @view D[1:10, 1:10]
			dsum = ApproxFunBase.summarystr(S)
			show(io, MIME"text/plain"(), S)
			@test contains(String(take!(io)), dsum)
		end
		@testset "QuotientSpace" begin
			Q = QuotientSpace(Dirichlet(ConstantSpace(0..1)))
			@test startswith(repr(Q), "ConstantSpace(0..1) /")
			io = IOBuffer()
			show(io, MIME"text/plain"(), Q)
			s = String(take!(io))
			@test startswith(s, "ConstantSpace(0..1) /")
		end
	end
end
