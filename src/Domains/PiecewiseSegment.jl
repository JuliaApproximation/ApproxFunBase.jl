export PiecewiseSegment

struct PiecewiseSegment{T,V<:AbstractVector{T}} <: Domain{T}
    points::V
end
PiecewiseSegment{T}(d::V) where {T,V<:AbstractVector{T}} = PiecewiseSegment{T,V}(d)
PiecewiseSegment(d...) = PiecewiseSegment(SVector{length(d), promote_eltypeof(d)}(d))

function PiecewiseSegment(pcsin::AbstractVector{IT}) where IT<:IntervalOrSegment
    pcs=collect(pcsin)
    p=∂(pop!(pcs))
    successful=true
    while successful
        successful=false
        for k=1:length(pcs)
            if leftendpoint(pcs[k]) == last(p)
                push!(p,rightendpoint(pcs[k]))
                deleteat!(pcs,k)
                successful=true
                break
            end
        end
    end
    @assert isempty(pcs)
    PiecewiseSegment(p)
end

==(a::PiecewiseSegment,b::PiecewiseSegment) = a.points==b.points

indomain(x, d::PiecewiseSegment) = any(Base.Fix1(in, x), components(d))


canonicaldomain(d::PiecewiseSegment)=d
ncomponents(d::PiecewiseSegment)=length(d.points)-1
component(d::PiecewiseSegment{T}, j::Integer) where {T} = Segment{T}(d.points[j], d.points[j+1])
components(d::PiecewiseSegment) = [component(d,k) for k=1:ncomponents(d)]
function components(d::PiecewiseSegment{<:Any, <:SVector})
    SVector(ntuple(k->component(d,k), ncomponents(d)))
end

for OP in (:arclength,:complexlength)
    @eval $OP(d::PiecewiseSegment) = mapreduce($OP,+,components(d))
end

isperiodic(_) = false
isperiodic(d::PiecewiseSegment) = first(d.points) == last(d.points)

reverseorientation(d::PiecewiseSegment) = PiecewiseSegment(reverse(d.points))

isambiguous(d::PiecewiseSegment) = isempty(d.points)
convert(::Type{PiecewiseSegment{T}},::AnyDomain) where {T<:Number} = PiecewiseSegment{T}([])
convert(::Type{IT},::AnyDomain) where {IT<:PiecewiseSegment}=PiecewiseSegment(Float64[])


function points(d::PiecewiseSegment,n)
   k=div(n,ncomponents(d))
    r=n-ncomponents(d)*k

    float(eltype(d))[vcat([points(component(d,j),k+1) for j=1:r]...);
        vcat([points(component(d,j),k) for j=r+1:ncomponents(d)]...)]
end



rand(d::PiecewiseSegment) = rand(d[rand(1:ncomponents(d))])
checkpoints(d::PiecewiseSegment{T}) where {T} = mapreduce(checkpoints,union,components(d))

leftendpoint(d::PiecewiseSegment) = first(d.points)
rightendpoint(d::PiecewiseSegment) = last(d.points)


# Comparison with UnionDomain
for OP in (:(isapprox),:(==))
    @eval begin
        $OP(a::PiecewiseSegment,b::UnionDomain) = $OP(UnionDomain(components(a)),b)
        $OP(b::UnionDomain,a::PiecewiseSegment) = $OP(UnionDomain(components(a)),b)
    end
end


function union(S::PiecewiseSegment{<:Real}, D::IntervalOrSegment{<:Real})
    isempty(D) && return S
    a,b = endpoints(D)
    (a ∈ S || b ∈ S) && return PiecewiseSegment(sort!(union(S.points, a, b)))
    UnionDomain(S, D)
end
union(D::IntervalOrSegment{<:Real}, S::PiecewiseSegment{<:Real}) = union(S,D)

for OP in (:minimum, :maximum)
    @eval $OP(d::PiecewiseSegment) = $OP(d.points)
end
