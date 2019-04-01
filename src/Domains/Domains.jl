include("Segment.jl")
include("UnionDomain.jl")
include("PiecewiseSegment.jl")
include("Point.jl")




convert(::Type{Space},d::ClosedInterval) = Space(Domain(d))

#issubset between domains

issubset(a::IntervalOrSegment{T}, b::PiecewiseSegment{T}) where {T<:Real} =
    a⊆Segment(first(b.points),last(b.points))


function setdiff(b::Segment,a::Point)
    if !(a ⊆ b)
        b
    elseif leftendpoint(b) == a.x  || rightendpoint(b) == a.x
        b
    else
        Segment(leftendpoint(b),a.x) ∪ Segment(a.x,rightendpoint(b))
    end
end

# sort



function Base.setdiff(d::SegmentDomain,p::Point)
    x = Number(p)
    (x ∉ d || x ≈ leftendpoint(d) || x ≈ rightendpoint(d)) && return d
    DifferenceDomain(d,p)
end



# multivariate domainxs

include("multivariate.jl")
