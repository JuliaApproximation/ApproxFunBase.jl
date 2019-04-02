export ToeplitzOperator, HankelOperator


mutable struct ToeplitzOperator{T<:Number} <: Operator{T}
    negative::Vector{T}
    nonnegative::Vector{T}
end


ToeplitzOperator(V::Vector{T},W::Vector{Q}) where {T<:Number,Q<:Number} =
    ToeplitzOperator{promote_type(T,Q)}(V,W)
ToeplitzOperator(V::AbstractVector,W::AbstractVector) =
    ToeplitzOperator(collect(V),collect(W))

convert(::Type{Operator{TT}},T::ToeplitzOperator) where {TT} =
    ToeplitzOperator(convert(Vector{TT},T.negative),convert(Vector{TT},T.nonnegative))

for op in (:(Base.real), :(Base.imag))
    @eval $op(T::ToeplitzOperator) = ToeplitzOperator($op(T.negative), $op(T.nonnegative))
end

function SymToeplitzOperator(V::Vector)
    W=V[2:end]
    V=copy(V)
    V[1]*=2
    ToeplitzOperator(W,V)
end

for OP in (:domainspace,:rangespace)
    @eval $OP(T::ToeplitzOperator) = ℓ⁰
end

getindex(T::ToeplitzOperator,k::Integer,j::Integer) =
    toeplitz_getindex(T.negative,T.nonnegative,k,j)

function toeplitz_getindex(negative::AbstractVector{T},nonnegative::AbstractVector{T},k::Integer,j::Integer) where T
    if 0<k-j≤length(negative)
        negative[k-j]
    elseif 0≤j-k≤length(nonnegative)-1
        nonnegative[j-k+1]
    else
        zero(T)
    end
end

function toeplitz_getindex(cfs::AbstractVector{T},k::Integer,j::Integer) where T
    if k==j && !isempty(cfs)
        2cfs[1]
    elseif 0<k-j≤length(cfs)-1
        cfs[k-j+1]
    elseif 0<j-k≤length(cfs)-1
        cfs[j-k+1]
    else
        zero(T)
    end
end

function BandedMatrix(S::SubOperator{T,ToeplitzOperator{T},Tuple{UnitRange{Int},UnitRange{Int}}}) where T
    ret = BandedMatrix(Zeros, S)

    kr,jr=parentindices(S)

    neg=parent(S).negative
    pos=parent(S).nonnegative

    toeplitz_axpy!(1.0,neg,pos,kr,jr,ret)
end



bandwidths(T::ToeplitzOperator)=(length(T.negative),length(T.nonnegative)-1)


# slice of a ToeplitzOPerator is a ToeplitzOperator

function Base.getindex(T::ToeplitzOperator,kr::InfRanges,jr::InfRanges)
    sh=first(jr)-first(kr)
    st=step(jr)
    @assert st==step(kr)
    if sh ≥0
        ToeplitzOperator([reverse!(T.nonnegative[1:sh]);T.negative],T.nonnegative[sh+1:st:end])
    else
        ToeplitzOperator(T.negative[-sh+1:st:end],[reverse!(T.negative[1:-sh]);T.nonnegative])
    end
end





## Hankel Operator


mutable struct HankelOperator{T<:Number} <: Operator{T}
    coefficients::Vector{T}
end

for OP in (:domainspace,:rangespace)
    @eval $OP(T::HankelOperator) = ℓ⁰
end

HankelOperator(V::AbstractVector)=HankelOperator(collect(V))

HankelOperator(f::Fun)=HankelOperator(f.coefficients)



@eval convert(::Type{Operator{TT}},T::HankelOperator) where {TT}=HankelOperator(convert(Vector{TT},T.coefficients))

function hankel_getindex(v::AbstractVector,k::Integer,j::Integer)
   if k+j-1 ≤ length(v)
        v[k+j-1]
    else
        zero(eltype(v))
    end
end

getindex(T::HankelOperator,k::Integer,j::Integer) =
    hankel_getindex(T.coefficients,k,j)


function BandedMatrix(S::SubOperator{T,HankelOperator{T},Tuple{UnitRange{Int},UnitRange{Int}}}) where T
    ret=BandedMatrix(Zeros, S)

    kr,jr=parentindices(S)
    cfs=parent(S).coefficients

    hankel_axpy!(1.0,cfs,kr,jr,ret)
end


bandwidths(T::HankelOperator) = (max(0,length(T.coefficients)-1),max(0,length(T.coefficients)-1))



## algebra


function Base.maximum(T::ToeplitzOperator)
    if isempty(T.negative)
        maximum(T.nonnegative)
    elseif isempty(T.nonnegative)
        maximum(T.negative)
    else
        max(maximum(T.negative),maximum(T.nonnegative))
    end
end

-(T::ToeplitzOperator)=ToeplitzOperator(-T.negative,-T.nonnegative)
*(c::Number,T::ToeplitzOperator)=ToeplitzOperator(c*T.negative,c*T.nonnegative)

-(H::HankelOperator)=HankelOperator(-H.coefficients)
*(c::Number,H::HankelOperator)=HankelOperator(c*H.coefficients)


## inv

function Base.inv(T::ToeplitzOperator)
    @assert length(T.nonnegative)==1
    ai=\(T,[1.0];maxlength=100000)
    ToeplitzOperator(ai[2:end],ai[1:1])
end


####
# Toeplitz/Hankel
####


# αn,α0,αp give the constant for the negative, diagonal and positive
# entries.  The usual case is 1,2,1
function toeplitz_axpy!(αn,α0,αp,neg,pos,kr,jr,ret)
    dat=ret.data
    m=size(dat,2)

    dg=diagindrow(ret,kr,jr)

    # diagonal
    if dg ≥ 1
        α0p=α0*pos[1]
        @simd for j=1:m
            @inbounds dat[dg,j]+=α0p
        end
    end

    # positive entries
    for k=2:min(length(pos),dg)
        αpp=αp*pos[k]
        @simd for j=1:m
            @inbounds dat[dg-k+1,j]+=αpp
        end
    end

    # negative entries
    for k=1:min(length(neg),size(dat,1)-dg)
        αnn=αn*neg[k]
        @simd for j=1:m
            @inbounds dat[dg+k,j]+=αnn
        end
    end

    ret
end

# this routine is for when we want a symmetric toeplitz
function sym_toeplitz_axpy!(αn,α0,αp,cfs,kr,jr,ret)
    dg=diagindrow(ret,kr,jr)

    dat=ret.data
    m=size(dat,2)
    # diagonal
    if dg ≥ 1
        α0p=α0*cfs[1]
        @simd for j=1:m
            @inbounds dat[dg,j]+=α0p
        end
    end

    # positive entries
    for k=2:min(length(cfs),dg)
        αpp=αp*cfs[k]
        @simd for j=1:m
            @inbounds dat[dg-k+1,j]+=αpp
        end
    end

    # negative entries
    for k=2:min(length(cfs),size(dat,1)-dg+1)
        αnn=αn*cfs[k]
        @simd for j=1:m
            @inbounds dat[dg+k-1,j]+=αnn
        end
    end

    ret
end

toeplitz_axpy!(α,neg,pos,kr,jr,ret) =
    toeplitz_axpy!(α,α,α,neg,pos,kr,jr,ret)

sym_toeplitz_axpy!(α0,αp,cfs,kr,jr,ret) =
    sym_toeplitz_axpy!(αp,α0,αp,cfs,kr,jr,ret)


function hankel_axpy!(α,cfs,kr,jr,ret)
    dat=ret.data

    st=stride(dat,2)-2
    mink=first(kr)+first(jr)-1

    N=length(dat)

    # dg gives the row corresponding to the diagonal of the original operator
    dg=diagindrow(ret,kr,jr)

    # we need the entry where the first entry is written to
    # this is going to be a shift of the diagonal of the true operator
    dg1=dg+first(kr)-first(jr)

    for k=mink:min(length(cfs),size(dat,1)+mink-dg1)
        dk=k-mink+dg1
        nk=k-mink
        αc=α*cfs[k]
        @simd for j=dk:st:min(dk+nk*st,N)
            @inbounds dat[j]+=αc
        end
    end

    ret
end