
recA(::Type{T},::Chebyshev,k) where {T} = k == 0 ? one(T) : 2one(T)
recB(::Type{T},::Chebyshev,_) where {T} = zero(T)
recC(::Type{T},::Chebyshev,k) where {T} = one(T)   # one(T) ensures we get correct type

recα(::Type{T},::Chebyshev,_) where {T} = zero(T)
recβ(::Type{T},::Chebyshev,k) where {T} = ifelse(k==1,one(T),one(T)/2)   # one(T) ensures we get correct type,ifelse ensures inlining
recγ(::Type{T},::Chebyshev,k) where {T} = one(T)/2   # one(T) ensures we get correct type





## Evaluation

Evaluation(S::MaybeNormalized{<:Chebyshev},x::Number,o::Integer) = ConcreteEvaluation(S,x,o)

function evaluatechebyshev(n::Integer,x::T) where T<:Number
    if n == 1
        [one(T)]
    else
        p = zeros(T,n)
        p[1] = one(T)
        p[2] = x
        twox = 2x

        for j=2:n-1
            p[j+1] = muladd(twox, p[j], -p[j-1])
        end

        p
    end
end

# We assume that x is already scaled to the canonical domain. S is unused here
function forwardrecurrence(::Type{T},S::Chebyshev,r::AbstractUnitRange{<:Integer},x::Number) where {T}
    @assert !isempty(r) && first(r) >= 0
    v = evaluatechebyshev(maximum(r)+1, T(x))
    first(r) == 0 ? v : v[r .+ 1]
end

function getindex(op::ConcreteEvaluation{<:Chebyshev{<:IntervalOrSegment},<:SpecialEvalPtType}, j::Integer)
    _getindex_eval_endpoints(op, evaluation_point(op), j)
end
function getindex(op::ConcreteEvaluation{<:Chebyshev{<:IntervalOrSegment},<:SpecialEvalPtType}, j::AbstractUnitRange)
    _getindex_eval_endpoints(op, evaluation_point(op), j)
end

function _getindex_eval_endpoints(op, x, j)
    if isleftendpoint(x)
        _getindex_eval_leftendpoint(op, x, j)
    else
        _getindex_eval_rightendpoint(op, x, j)
    end
end
function _getindex_eval_leftendpoint(op::ConcreteEvaluation{<:Chebyshev{<:IntervalOrSegment}}, x, j::Integer)
    T=eltype(op)
    if op.order == 0
        ifelse(isodd(j),  # right rule
            one(T),
            -one(T))
    else
        #TODO: Fast version
        op[j:j][1]
    end
end
function _getindex_eval_rightendpoint(op::ConcreteEvaluation{<:Chebyshev{<:IntervalOrSegment}}, x, j::Integer)
    T=eltype(op)
    if op.order == 0
        one(T)
    else
        #TODO: Fast version
        op[j:j][1]
    end
end
function _getindex_eval_leftendpoint(op::ConcreteEvaluation{<:Chebyshev{<:IntervalOrSegment}}, x, k::AbstractUnitRange)
    Base.require_one_based_indexing(k)
    T=eltype(op)
    x = op.x
    d = domain(op)
    p = op.order
    cst = strictconvert(T,(2/complexlength(d))^p)
    n=length(k)

    ret = Array{T}(undef, n)
    for (ind, j) in enumerate(k)
        ret[ind] = iseven(p+j) ? -1 : 1
    end

    for m in 0:p-1
        for (ind, j) in enumerate(k)
            ret[ind] *= (j-1)^2-m^2
        end
        scal!(strictconvert(T,1/(2m+1)), ret)
    end

    scal!(cst,ret)
end
function _getindex_eval_rightendpoint(op::ConcreteEvaluation{<:Chebyshev{<:IntervalOrSegment}}, x, k::AbstractUnitRange)
    Base.require_one_based_indexing(k)
    T=eltype(op)
    x = op.x
    d = domain(op)
    p = op.order
    cst = strictconvert(T,(2/complexlength(d))^p)
    n=length(k)

    ret = fill(one(T),n)

    for m in 0:p-1
        for (ind, j) in enumerate(k)
            ret[ind] *= (j-1)^2-m^2
        end
        scal!(strictconvert(T,1/(2m+1)), ret)
    end

    scal!(cst,ret)
end

@inline function _Dirichlet_Chebyshev(S, order)
    order == 0 && return ConcreteDirichlet(S,
        ArraySpace(convert_vector_or_svector(
            ConstantSpace.(Point.(endpoints(domain(S)))))),
        0)
    default_Dirichlet(S,order)
end
@static if VERSION >= v"1.8"
    Base.@constprop :aggressive function Dirichlet(S::Chebyshev, order)
        _Dirichlet_Chebyshev(S, order)
    end
else
    function Dirichlet(S::Chebyshev, order)
        _Dirichlet_Chebyshev(S, order)
    end
end

function getindex(op::ConcreteDirichlet{<:Chebyshev},
                                             k::Integer,j::Integer)
    if op.order == 0
        k == 1 && iseven(j) && return -one(eltype(op))
        return one(eltype(op))
    else
        error("Only zero Dirichlet conditions implemented")
    end
end

function Matrix(S::SubOperator{T,ConcreteDirichlet{C,V,T},
                                NTuple{2,UnitRange{Int}}}) where {C<:Chebyshev,V,T}
    ret = Array{T}(undef, size(S)...)
    kr,jr = parentindices(S)
    isempty(kr) && return ret
    isempty(jr) && return ret
    if first(kr) == 1
        if isodd(jr[1])
            ret[1,1:2:end] .= one(T)
            ret[1,2:2:end] .= -one(T)
        else
            ret[1,1:2:end] .= -one(T)
            ret[1,2:2:end] .= one(T)
        end
    end
    if last(kr) == 2
        ret[end,:] .= one(T)
    end
    return ret
end
#

# Multiplication

Base.stride(M::ConcreteMultiplication{U,V}) where {U<:Chebyshev,V<:Chebyshev} =
    stride(M.f)

getindex(M::ConcreteMultiplication{C,C,T},k::Integer,j::Integer) where {T,C<:Chebyshev} =
    chebmult_getindex(coefficients(M.f),k,j)

getindex(M::ConcreteMultiplication{C,PS,T},k::Integer,j::Integer) where {PS<:PolynomialSpace,T,C<:Chebyshev} =
    M[k:k,j:j][1,1]


function BandedMatrix(S::SubOperator{T,ConcreteMultiplication{C,C,T},NTuple{2,UnitRange{Int}}}) where {C<:Chebyshev,T}
    ret = BandedMatrix(Zeros, S)

    kr,jr=parentindices(S)
    cfs=parent(S).f.coefficients

    isempty(cfs) && return ret

    # Toeplitz part
    sym_toeplitz_axpy!(1.0,0.5,cfs,kr,jr,ret)

    #Hankel part
    hankel_axpy!(0.5,cfs,kr,jr,ret)

    # divide first row by half
    if first(kr)==1
        if first(jr)==1
            ret[1,1]+=0.5cfs[1]
        end

        for j=1:min(1+ret.u,size(ret,2))
            ret[1,j]/=2
        end
    end


    ret
end



## Derivative

function Derivative(sp::Chebyshev{DD},order::Number) where {DD<:IntervalOrSegment}
    assert_integer(order)
    ConcreteDerivative(sp,order)
end


rangespace(D::ConcreteDerivative{Chebyshev{DD,RR}}) where {DD<:IntervalOrSegment,RR} =
    Ultraspherical(D.order,domain(D))

bandwidths(D::ConcreteDerivative{Chebyshev{DD,RR}}) where {DD<:IntervalOrSegment,RR} = -D.order,D.order
Base.stride(D::ConcreteDerivative{Chebyshev{DD,RR}}) where {DD<:IntervalOrSegment,RR} = D.order

isdiag(D::ConcreteDerivative{<:Chebyshev{<:IntervalOrSegment}}) = false

function getindex(D::ConcreteDerivative{Chebyshev{DD,RR},K,T},k::Integer,j::Integer) where {DD<:IntervalOrSegment,RR,K,T}
    m=D.order
    d=domain(D)

    if j==k+m
        C=strictconvert(T,pochhammer(one(T),m-1)/2*(4/complexlength(d))^m)
        strictconvert(T,C*(m+k-one(T)))
    else
        zero(T)
    end
end

linesum(f::Fun{Chebyshev{DD,RR}}) where {DD<:IntervalOrSegment,RR} =
    sum(setcanonicaldomain(f))*arclength(domain(f))/2



## Clenshaw-Curtis functional

for (Func,Len) in ((:DefiniteIntegral,:complexlength),(:DefiniteLineIntegral,:arclength))
    ConcFunc = Symbol(:Concrete, Func)
    @eval begin
        $Func(S::Chebyshev{D}) where {D<:IntervalOrSegment} = $ConcFunc(S)
        function getindex(Σ::$ConcFunc{Chebyshev{D,R},T},k::Integer) where {D<:IntervalOrSegment,R,T}
            d = domain(Σ)
            C = $Len(d)/2

            isodd(k) ? strictconvert(T,2C/(k*(2-k))) : zero(T)
        end
    end
end



ReverseOrientation(S::Chebyshev) = ReverseOrientationWrapper(NegateEven(S,reverseorientation(S)))
Reverse(S::Chebyshev) = ReverseWrapper(NegateEven(S,S))
