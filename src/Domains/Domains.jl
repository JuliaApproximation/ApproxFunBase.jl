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

function affine_setdiff(d::Domain, ptsin::UnionDomain)    
    pts=Number.(elements(ptsin))
    isempty(pts) && return d
    tol=sqrt(eps(arclength(d)))
    da=leftendpoint(d)
    isapprox(da,pts[1];atol=tol) && popfirst!(pts)
    isempty(pts) && return d
    db=rightendpoint(d)
    isapprox(db,pts[end];atol=tol) && pop!(pts)

    sort!(pts)
    leftendpoint(d) > rightendpoint(d) && reverse!(pts)
    filter!(p->p ∈ d,pts)

    isempty(pts) && return d
    length(pts) == 1 && return d \ pts[1]

    ret = Array{Domain}(undef, length(pts)+1)
    ret[1] = Domain(leftendpoint(d) .. pts[1])
    for k = 2:length(pts)
        ret[k] = Domain(pts[k-1]..pts[k])
    end
    ret[end] = Domain(pts[end] .. rightendpoint(d))
    UnionDomain(ret)
end