export OperatorFunction


abstract type OperatorFunction{BT,FF,T} <: Operator{T} end

struct ConcreteOperatorFunction{BT<:Operator,FF,T} <: OperatorFunction{BT,FF,T}
    op::BT
    f::FF
end

function ConcreteOperatorFunction(op::Operator,f::Function)
    T = typeof(f(oneunit(eltype(op))))
    ConcreteOperatorFunction{typeof(op),typeof(f),T}(op,f)
end
OperatorFunction(op::Operator,f::Function) = ConcreteOperatorFunction(op,f)

for op in (:domainspace,:rangespace,:domain,:bandwidths)
    @eval begin
        $op(OF::ConcreteOperatorFunction) = $op(OF.op)
    end
end

function getindex(OF::ConcreteOperatorFunction,k::Integer,j::Integer)
    @assert isdiag(OF.op)
    if k==j
        OF.f(OF.op[k,k])::eltype(OF)
    else
        zero(eltype(OF))
    end
end

function convert(::Type{Operator{T}},D::ConcreteOperatorFunction) where T
    if T==eltype(D)
        D
    else
        DopT = strictconvert(Operator{T}, D.op)
        ConcreteOperatorFunction(DopT, D.f)::Operator{T}
    end
end


for OP in (:(Base.inv),:(Base.sqrt))
    @eval begin
        $OP(D::DiagonalOperator) = OperatorFunction(D,$OP)
        $OP(C::ConstantTimesOperator) = $OP(C.λ)*$OP(C.op)
        function $OP(D::ConcreteOperatorFunction)
            @assert isdiag(D)
            OperatorFunction(D.op,x->$OP(D.f(x)))
        end
        $OP(A::Operator) = isdiag(A) ? OperatorFunction(A,$OP) :
                                error("Not implemented.")
    end
end


Base.sqrt(S::SpaceOperator) = SpaceOperator(sqrt(S.op),S.domainspace,S.rangespace)
Base.inv(S::SpaceOperator) = SpaceOperator(inv(S.op),S.rangespace,S.domainspace)
