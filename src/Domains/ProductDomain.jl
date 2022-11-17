
nfactors(d::ProductDomain) = length(components(d))
factors(d::ProductDomain) = components(d)
factor(d::ProductDomain,k::Integer) = component(d,k)

canonicaldomain(d::ProductDomain) = ProductDomain(map(canonicaldomain,factors(d))...)


# product domains are their own canonical domain
for OP in (:fromcanonical,:tocanonical)
    @eval begin
        $OP(d::ProductDomain, x::SVector) = SVector(map($OP,factors(d),x)...)
        $OP(d::ProductDomain, x::SVector{2}) = SVector($OP(first(factors(d)), first(x)), $OP(last(factors(d)), last(x)))
    end
end



function pushappendpts!(ret, xx, pts)
    if isempty(pts)
        push!(ret,SVector(xx...))
    else
        for x in pts[1]
            pushappendpts!(ret,(xx...,x...),pts[2:end])
        end
    end
    ret
end

function checkpoints(d::ProductDomain)
    pts = checkpoints.(factors(d))
    ret=Vector{SVector{sum(dimension.(factors(d))),float(promote_type(eltype.(eltype.(factors(d)))...))}}(undef, 0)

    pushappendpts!(ret,(),pts)
    ret
end

function points(d::ProductDomain,n::Tuple)
    @assert length(factors(d)) == length(n)
    pts=map(points,factors(d),n)
    ret=Vector{SVector{length(factors(d)),mapreduce(eltype,promote_type,factors(d))}}(undef, 0)
    pushappendpts!(ret,SVector(x),pts)
    ret
end

reverseorientation(d::ProductDomain) = ProductDomain(map(reverseorientation, factors(d)))

domainscompatible(a::ProductDomain,b::ProductDomain) =
                        length(factors(a))==length(factors(b)) &&
                        all(map(domainscompatible,factors(a),factors(b)))
