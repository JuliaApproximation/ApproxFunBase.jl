@testset "show" begin
	io = IOBuffer()
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
			summarystr = ApproxFunBase.summarystr(D)
			@test repr(D) == summarystr
			show(io, MIME"text/plain"(), D)
			@test contains(String(take!(io)), summarystr)

			D = Derivative(Chebyshev())
			summarystr = ApproxFunBase.summarystr(D)
			show(io, MIME"text/plain"(), D)
			@test contains(String(take!(io)), summarystr)
		end
		@testset "SubOperator" begin
			D = Derivative(Chebyshev())
			S = @view D[1:10, 1:10]
			summarystr = ApproxFunBase.summarystr(S)
			show(io, MIME"text/plain"(), S)
			@test contains(String(take!(io)), summarystr)
		end
		@testset "Evaluation" begin
			E = Evaluation(Chebyshev(), 0)
			summarystr = ApproxFunBase.summarystr(E)
			show(io, MIME"text/plain"(), E)
			@test contains(String(take!(io)), summarystr)

			EA = Evaluation(Chebyshev(), 0)'
			summarystr = ApproxFunBase.summarystr(EA)
			show(io, MIME"text/plain"(), EA)
			@test contains(String(take!(io)), summarystr)

			EA = transpose(Evaluation(Chebyshev(), 0))
			summarystr = ApproxFunBase.summarystr(EA)
			show(io, MIME"text/plain"(), EA)
			@test contains(String(take!(io)), summarystr)
		end
		@testset "QuotientSpace" begin
			Q = QuotientSpace(Dirichlet(ConstantSpace(0..1)))
			@test startswith(repr(Q), "ConstantSpace(0..1) /")
			show(io, MIME"text/plain"(), Q)
			s = String(take!(io))
			@test startswith(s, "ConstantSpace(0..1) /")
		end
	end
end
