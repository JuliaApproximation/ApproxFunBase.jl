

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

convert(::Type{Operator{T}},A::PartialInverseOperator) where {T}=PartialInverseOperator(convert(Operator{T},A.cache), A.bandwidths)

domainspace(P::PartialInverseOperator)=rangespace(P.cache)
rangespace(P::PartialInverseOperator)=domainspace(P.cache)
domain(P::PartialInverseOperator)=domain(domainspace(P))
bandwidths(P::PartialInverseOperator) = P.bandwidths

function getindex(P::PartialInverseOperator,k::Integer,j::Integer)
    b = bandwidth(P.cache, 2)
    if k == j
        return inv(P.cache[k,k])
    elseif j > k
        t = zero(T)
        for i = max(k,j-b-1):j-1
            t += ret[k,i]*P.cache[i,j]
        end
        return -t/P.cache[j,j]
    else
        return zero(eltype(P))
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


