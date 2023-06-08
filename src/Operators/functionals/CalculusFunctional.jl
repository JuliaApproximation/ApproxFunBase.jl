export DefiniteIntegral,DefiniteLineIntegral

abstract type CalculusFunctional{S,T} <: Operator{T} end

@functional CalculusFunctional

##TODO: Add ConcreteOp

macro calculus_functional(Op)
    ConcOp = Symbol(:Concrete, Op)
    WrappOp = Symbol(Op, :Wrapper)
    return esc(quote
        abstract type $Op{SSS,TTT} <: ApproxFunBase.CalculusFunctional{SSS,TTT} end
        struct $ConcOp{S,T} <: $Op{S,T}
            domainspace::S
        end
        struct $WrappOp{BT<:Operator,S<:Space,T} <: $Op{S,T}
            op::BT
            domainspace::S
        end

        ApproxFunBase.@wrapper $WrappOp false false
        ApproxFunBase.domainspace(A::$WrappOp) = A.domainspace

        # We expect the operator to be real/complex if the basis is real/complex
        $ConcOp(dsp::ApproxFunBase.Space) = $ConcOp{typeof(dsp),ApproxFunBase.prectype(dsp)}(dsp)

        $Op() = $Op(ApproxFunBase.UnsetSpace())
        $Op(dsp) = $ConcOp(dsp)
        $Op(d::ApproxFunBase.Domain) = $Op(ApproxFunBase.Space(d))

        ApproxFunBase.promotedomainspace(::$Op,sp::ApproxFunBase.Space) = $Op(sp)


        Base.convert(::Type{ApproxFunBase.Operator{T}},Σ::$ConcOp) where {T} =
            (T==eltype(Σ) ? Σ : $ConcOp{typeof(Σ.domainspace),T}(Σ.domainspace))::ApproxFunBase.Operator{T}

        ApproxFunBase.domain(Σ::$ConcOp) = ApproxFunBase.domain(Σ.domainspace)
        ApproxFunBase.domainspace(Σ::$ConcOp) = Σ.domainspace

        Base.getindex(::$ConcOp{ApproxFunBase.UnsetSpace},kr::AbstractRange) =
            error("Spaces cannot be inferred for operator")

        $WrappOp(op::ApproxFunBase.Operator, dsp = ApproxFunBase.domainspace(op)) =
            $WrappOp{typeof(op),typeof(dsp),eltype(op)}(op, dsp)


        function Base.convert(::Type{ApproxFunBase.Operator{T}},Σ::$WrappOp) where {T}
            T==eltype(Σ) && return Σ
            $WrappOp(ApproxFunBase.strictconvert(ApproxFunBase.Operator{T},Σ.op))::ApproxFunBase.Operator{T}
        end
    end)
end

@calculus_functional(DefiniteIntegral)
@calculus_functional(DefiniteLineIntegral)


#default implementation

DefiniteIntegral(sp::UnsetSpace) = ConcreteDefiniteIntegral(sp)
DefiniteLineIntegral(sp::UnsetSpace) = ConcreteDefiniteLineIntegral(sp)

"""
    DefiniteIntegral([sp::Space])

Return the operator that integrates a `Fun` over its domain. If `sp` is unspecified,
it is inferred at runtime from the context.

# Examples
```jldoctest
julia> f = Fun(x -> 3x^2, Chebyshev());

julia> DefiniteIntegral() * f ≈ 2 # integral of 3x^2 over -1..1
true
```
"""
function DefiniteIntegral(sp::Space)
    if typeof(canonicaldomain(sp)) == typeof(domain(sp))
        # try using `Integral`
        Q = Integral(sp)
        rsp = rangespace(Q)
        TE = eltype(Evaluation(rsp,rightendpoint))
        A = (Evaluation(rsp,rightendpoint)-Evaluation(rsp,leftendpoint)) * Q
        S = SpaceOperator(A, sp, ConstantSpace(promote_type(TE, eltype(Q))))
        DefiniteIntegralWrapper(S)
    else
        # try mapping to canonical domain
        M = Multiplication(fromcanonicalD(sp),setcanonicaldomain(sp))
        Op = DefiniteIntegral(rangespace(M))*M
        DefiniteIntegralWrapper(SpaceOperator(Op,sp,rangespace(Op)))
    end
end

function DefiniteLineIntegral(sp::Space)
    if typeof(canonicaldomain(sp)) == typeof(domain(sp))
        error("Override DefiniteLineIntegral for $sp")
    end

    M = Multiplication(abs(fromcanonicalD(sp)),setcanonicaldomain(sp))
    Op = DefiniteLineIntegral(rangespace(M))*M
    DefiniteLineIntegralWrapper(SpaceOperator(Op,sp,rangespace(Op)))
end


#TODO: Remove SPECIALOPS reimplement
# *{T,D<:DefiniteIntegral,M<:Multiplication}(A::TimesFunctional{T,D,M},b::Fun) = bilinearform(A.op.f,b)
# *{T,D<:DefiniteLineIntegral,M<:Multiplication}(A::TimesFunctional{T,D,M},b::Fun) = linebilinearform(A.op.f,b)
# *{T,D<:Union{DefiniteIntegral,DefiniteLineIntegral},
#   M<:Multiplication,V}(A::FunctionalOperator{TimesFunctional{T,D,M},V},b::Fun) =
#     Fun(A.op*b)
