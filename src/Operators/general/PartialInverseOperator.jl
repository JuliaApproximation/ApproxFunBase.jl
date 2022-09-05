export PartialInverseOperator


struct PartialInverseOperator{T<:Number,CO<:CachedOperator,BI} <: Operator{T}
    cache::CO
    bandwidths::BI
end

function PartialInverseOperator(CO::CachedOperator{T},bandwidths) where T<:Number
    @assert istriu(CO) # || istril(CO)
    return PartialInverseOperator{T,typeof(CO),typeof(bandwidths)}(CO,bandwidths)
end

PartialInverseOperator(B::Operator, bandwidths) = PartialInverseOperator(cache(B), bandwidths)
PartialInverseOperator(B::Operator) = PartialInverseOperator(B, bandwidths(B))

convert(::Type{Operator{T}},A::PartialInverseOperator) where {T}=PartialInverseOperator(strictconvert(Operator{T},A.cache), A.bandwidths)

domainspace(P::PartialInverseOperator)=rangespace(P.cache)
rangespace(P::PartialInverseOperator)=domainspace(P.cache)
domain(P::PartialInverseOperator)=domain(domainspace(P))
bandwidths(P::PartialInverseOperator) = P.bandwidths

# Compute the value at the (k,j)th index of inv(C), assumming that C is upper triangular
function _getindexinv(C, k::Integer, j::Integer, ::Type{UpperTriangular})
    j >= k || return zero(inv(one(eltype(C))))
    j == k && return inv(C[k,k])
    t = inv(C[k,k])
    # k-th row of the inverse, starting from the diagonal
    # but leaving out the j-th column
    ret = zeros(eltype(t), j-k)
    ret[1] = t
    for m in k+1:j-1 # populate the k-th row of the inverse
        t = zero(eltype(ret))
        for i in k:j-1
            t -= ret[i-k+1] * C[i,m]
        end
        ret[m - k + 1] = t/C[m,m]
    end
    t = zero(eltype(ret))
    for (rind, i) in enumerate(k:j-1)
        t -= ret[rind] * C[i,j]
    end
    t/C[j,j]
end

function getindex(P::PartialInverseOperator,k::Integer,j::Integer)
    b = bandwidth(P, 2)
    if k == j
        inv(P.cache[k,k])
    elseif j > k + b + 1
        zero(eltype(P))
    elseif j > k
        _getindexinv(P.cache, k, j, UpperTriangular)
    else
        zero(eltype(P))
    end
end


## These are both hacks that apparently work

function BandedMatrix(S::SubOperator{T,PP,Tuple{UnitRange{Int},UnitRange{Int}}}) where {T,PP<:PartialInverseOperator}
    kr,jr = parentindices(S)
    P = parent(S)
    #ret = BandedMatrix{eltype(S)}(undef, size(S), bandwidths(S))
    ret = BandedMatrix{eltype(S)}(undef, (last(kr),last(jr)), bandwidths(P))
    b = bandwidth(P, 2)
    #@assert first(kr) == first(jr) == 1

    @inbounds for j in 1:last(jr)
        kk = colrange(ret, j)
        if j in kk
            ret[j,j] = inv(P.cache[j,j])
        end
        for k in first(kk):min(last(kk),j-1)
            t = zero(T)
            for i = max(k,j-b-1):j-1
                t += ret[k,i]*P.cache[i,j]
            end
            ret[k,j] = -t/P.cache[j,j]
        end
    end

    ret[kr,jr]
end


