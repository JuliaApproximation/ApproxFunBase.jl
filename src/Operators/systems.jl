##Operators
# TODO: REMOVE!
for op in (:Derivative,:Integral)
    @eval begin
        function ($op)(d::AbstractVector{T}) where T<:IntervalOrSegment
            n=length(d)
            R=zeros(Operator{promote_eltypeof(d)},n,n)
            for k=1:n
                R[k,k]=$op(d[k])
            end

            R
        end
    end
end

function Evaluation(d::AbstractVector{T},x...) where T<:IntervalOrSegment
    n=length(d)
    R=zeros(Operator{promote_eltypeof(d)},n,n)
    for k=1:n
        R[k,k]=Evaluation(d[k],x...)
    end

    R
end


## Construction
function diagm_container(size, kv::Pair{<:Integer,<:AbstractVector{<:Operator}}...)
    T = mapreduce(x -> promote_eltypeof(x.second),
                    promote_type, kv)
    n = mapreduce(x -> length(x.second) + abs(x.first), max, kv)
    zeros(Operator{T}, n, n)
end

## broadcase

broadcast(::typeof(*),A::AbstractArray{N},D::Operator) where {N<:Number} =
    Operator{promote_type(N,eltype(D))}[A[k,j]*D for k=1:size(A,1),j=1:size(A,2)]
broadcast(::typeof(*),D::Operator,A::AbstractArray{N}) where {N<:Number}=A.*D
