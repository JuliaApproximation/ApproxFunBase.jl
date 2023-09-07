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
			@test startswith(repr(c), "ConstantSpace")
			@test contains(repr(c), repr(domain(c)))
		end
		@testset "TensorSpace" begin
			S1 = PointSpace(1:3)
			S = S1 ⊗ S1
			v = strip.(split(repr(S), '⊗'))
			@test length(v) == 2
			@test all(==(repr(S1)), v)
		end
		@testset "ProductSpace" begin
			S1 = PointSpace(1:4)
			S2 = PointSpace(1:2)
			P = ProductSpace([S1, S1], S2)
			@test startswith(repr(P), "ProductSpace")
			@test contains(repr(P), string(factors(P)))
		end
		@testset "SumSpace" begin
			S1 = PointSpace(1:3)
			S = S1 ⊕ S1
			v = strip.(split(repr(S), '⊕'))
			@test length(v) == 2
			@test all(==(repr(S1)), v)
		end
		@testset "PiecewiseSpace" begin
			p = PointSpace(1:4)
			ps = PiecewiseSpace(p)
			rpr = repr(ps)
			@test startswith(rpr, "PiecewiseSpace")
			@test contains(rpr, repr(p))
		end
		@testset "ArraySpace" begin
			spaces = [PointSpace(1:1), PointSpace(2:2)]
			A = ApproxFunBase.ArraySpace(spaces)
			@test startswith(repr(A), "$(ApproxFunBase.ArraySpace)")
			@test contains(repr(A), repr(spaces))
		end
	end
	@testset "Fun" begin
		f = Fun(PointSpace(1:3), [1,2,3])
		s = repr(f)
		@test startswith(s, "Fun")
		@test contains(s, repr(space(f)))
		@test contains(s, repr(coefficients(f)))

		f = Fun(ConstantSpace(0..1), [2])
		@test contains(repr(f), repr(f(0)))
		@test contains(repr(f), repr(domain(f)))

		f = Fun(ConstantSpace(), [2])
		@test contains(repr(f), repr(f(0)))

		f = Fun(ApproxFunBase.ArraySpace([ConstantSpace(0..1)]), [3])
		@test contains(repr(f), repr(f(0)))

		f = Fun(1, ConstantSpace(Point(3)))
		@test contains(repr(f), repr(domain(f)))
		@test contains(repr(f), repr(1))

		f = Fun(1, ConstantSpace(Point(3) ∪ Point(4)))
		@test contains(repr(f), repr(domain(f)))
		@test contains(repr(f), repr(1))
	end
	@testset "Operator" begin
		@testset "QuotientSpace" begin
			Q = QuotientSpace(Dirichlet(ConstantSpace(0..1)))
			@test startswith(repr(Q), "ConstantSpace($(0..1)) /")
			show(io, MIME"text/plain"(), Q)
			s = String(take!(io))
			@test startswith(s, "ConstantSpace($(0..1)) /")
		end
		@testset "ConstantOperator" begin
			A = I : PointSpace(1:4)
			s = summary(A)
			@test startswith(s, "ConstantOperator")
		end
		@testset "compact show" begin
			p  = [1+1e-8];
			M = Multiplication(Fun(PointSpace(p)), PointSpace(p));
			s = sprint(show, MIME"text/plain"(), M)
			@test s == "FiniteOperator : PointSpace([1.00000001]) → PointSpace([1.00000001])\n 1.00000001"
			p  = [1+1e-8, 2+1e-8];
			M = Multiplication(Fun(PointSpace(p)), PointSpace(p));
			s = sprint(show, MIME"text/plain"(), M)
			@test s == ("FiniteOperator : PointSpace([1.00000001, 2.00000001]) → PointSpace([1.00000001, 2.00000001])"*
							"\n 1.0   ⋅ \n  ⋅   2.0")
		end
	end
	@testset "Iterators" begin
		t = ([1,1], [1,1])
		B = ApproxFunBase.BlockInterlacer(t)
		@test repr(B) == "$(ApproxFunBase.BlockInterlacer)($(repr(t)))"
		C = cache(B)
		@test contains(repr(C), "Cached " * repr(B))
	end
	@testset "Tensorizer" begin
		o = Ones(Int,ℵ₀)
		t = ApproxFunBase.Tensorizer((o,o))
		@test repr(t) == "ApproxFunBase.Tensorizer($((o,o)))"
	end
end
