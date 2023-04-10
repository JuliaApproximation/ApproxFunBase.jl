
valsdomain_type_promote(::Type{T},::Type{T}) where {T<:Complex}=T,T
valsdomain_type_promote(::Type{T},::Type{T}) where {T<:Real}=T,T
valsdomain_type_promote(::Type{Int},::Type{Int})=Float64,Int
valsdomain_type_promote(::Type{T},::Type{Complex{V}}) where {T<:Real,V<:Real}=promote_type(T,V),Complex{promote_type(T,V)}
valsdomain_type_promote(::Type{Complex{T}},::Type{V}) where {T<:Real,V<:Real}=Complex{promote_type(T,V)},promote_type(T,V)
valsdomain_type_promote(::Type{T},::Type{Int}) where {T<:Integer}=Float64,Int
valsdomain_type_promote(::Type{T},::Type{Int}) where {T<:Real}=T,Int
valsdomain_type_promote(::Type{T},::Type{Int}) where {T<:Complex}=T,Int
valsdomain_type_promote(::Type{T},::Type{V}) where {T<:Integer,V<:Real}=valsdomain_type_promote(Float64,V)
valsdomain_type_promote(::Type{T},::Type{V}) where {T<:Integer,V<:Complex}=valsdomain_type_promote(Float64,V)
valsdomain_type_promote(::Type{T},::Type{Vector{T}}) where {T<:Real}=T,Vector{T}
valsdomain_type_promote(::Type{T},::Type{V}) where {T,V}=promote_type(T,V),promote_type(T,V)



function choosefuncfstype(ftype,Td)
    if !( ftype<: Number || ( ((ftype <: AbstractArray) || (ftype <: SVector)) &&
                              (eltype(ftype) <: Number) ) )
        @warn "Function outputs type $(ftype), which is not a Number"
    end

    Tprom = ftype

    if ftype <: Number #TODO should also work for array-valued functions
        Tprom,Tpromd=valsdomain_type_promote(ftype,Td)

        if Tpromd != Td
                @warn "Space domain number type $(Td) is not compatible with coefficient type $(Tprom)"
                #TODO should construct a new Space that contains a domain where the numbers have been promoted
                #and call constructor with this Space.
        end
    end

    Tprom
end


# default_Fun is the default constructor, based on evaluation and transforms
# last argument is whether to splat or not
function default_Fun(T::Type, f, d::Space, pts::AbstractArray, shouldsplat::Val{true})
    default_Fun(T, Base.splat(f), d, pts, Val(false))
end

function default_Fun(T::Type, f, d::Space, pts::AbstractArray, shouldsplat::Val{false})
    fv = broadcast!(f, similar(pts, T), pts)
    tfn = _transform!!(Val(supportsinplacetransform(d)))
    coeffs = tfn(d, fv)
    Fun(d, coeffs)
end


function default_Fun(f, d::Space, n::Integer, shouldsplat::Val{false})
    pts=points(d, n)
    f1=f(pts[1])
    if isa(f1,AbstractArray) && size(d) ≠ size(f1)
        return Fun(f,Space(fill(d,size(f1))),n)
    end

    # we need 3 eltype calls for the case Interval(Point([1.,1.]))
    Tprom=choosefuncfstype(typeof(f1),prectype(domain(d)))
    default_Fun(Tprom,f,d,pts,Val(false))
end

function default_Fun(f, d::Space, n::Integer, shouldsplat::Val{true})
    default_Fun(Base.splat(f), d, n, Val(false))
end

default_Fun(f,d::Space,n::Integer) = default_Fun(f,d,n,Val(!hasonearg(f)))

Fun(f,d::Space,n::Integer) = default_Fun(f,d,n)

# the following is to avoid ambiguity
# Fun(f::Fun,d) should be equivalent to Fun(x->f(x),d)
Fun(f::Fun,d::Space) = Fun(d,coefficients(f,d))
Fun(f::Fun,::Type{T}) where {T<:Space} = Fun(f,T(domain(f)))


Fun(f,T::Type) = Fun(f,T())
Fun(f,T::Type,n::Integer) = Fun(f,T(),n)

Fun(f::AbstractVector,d::Domain) = Fun(f,Space(d))
Fun(d::Domain,f::AbstractVector) = Fun(Space(d),f)


Fun(f,d::Domain,n) = Fun(f,Space(d),n)


# We do zero special since zero exists even when one doesn't
Fun(c::Number,::Type{T}) where {T<:Space} = c==0 ? zeros(T(AnyDomain())) : c*ones(T(AnyDomain()))
Fun(c::Number,d::Domain) = c==0 ? c*zeros(d) : c*ones(d)
Fun(c::Number,d::Space) = c==0 ? c*zeros(prectype(d),d) : c*ones(prectype(d),d)

## Adaptive constructors
function default_Fun(f, d::Space)
    _default_Fun(hasonearg(f) ? f : Base.splat(f), d)
end
# In _default_Fun, we know that the function takes a single argument
function _default_Fun(f, d::Space)
    isinf(dimension(d)) || return Fun(f,d,dimension(d))  # use exactly dimension number of sample points

    #TODO: reuse function values?
    T = real(prectype(domain(d)))

    r=checkpoints(d)
    f0=f(first(r))

    isa(f0,AbstractArray) && size(d) ≠ size(f0) && return Fun(f,Space(fill(d,size(f0))))

    tol =T==Any ? 20eps() : 20eps(T)

    fr=map(f,r)
    maxabsfr=norm(fr,Inf)

    for logn = 4:20
        #cf = Fun(f, d, 2^logn + 1)
        cf = default_Fun(f, d, 2^logn, Val(false))
        maxabsc = maximum(abs,cf.coefficients)
        if maxabsc == 0 && maxabsfr == 0
            return zeros(d)
        end

        b = block(d,length(cf.coefficients))
        bs = blockstart(d,max(b.n[1]-2,1))

        # we allow for transformed coefficients being a different size
        ##TODO: how to do scaling for unnormalized bases like Jacobi?
        if ncoefficients(cf) > 8 && maximum(abs, @view cf.coefficients[bs:end]) < 10tol*maxabsc &&
                all(k->norm(cf(r[k])-fr[k],1)<tol*length(cf.coefficients)*maxabsfr*1000,1:length(r))
            return chop!(cf,tol)
        end
    end

    @warn "Maximum number of coefficients "*string(2^20+1)*" reached in constructing Fun."

    Fun(f,d,2^21)
end

Fun(f::Type, d::Space) = error("Not implemented")


# special case constructors
"""
    zeros(d::Space)

Return the `Fun` that represents the function one on the specified space.

# Examples
```jldoctest
julia> zeros(Chebyshev())
Fun(Chebyshev(), [0.0])
```
"""
zeros(S::Space) = zeros(Float64, S)
zeros(::Type{T}, S::Space) where {T<:Number} = Fun(S,zeros(T,1))

# catch all
"""
    ones(d::Space)

Return the `Fun` that represents the function one on the specified space.

# Examples
```jldoctest
julia> ones(Chebyshev())
Fun(Chebyshev(), [1.0])
```
"""
ones(S::Space) = ones(Float64, S)
ones(::Type{T}, S::Space) where {T<:Number} = Fun(x->one(T),S)

function Fun(::typeof(identity), d::Domain)
    cd=canonicaldomain(d)
    if typeof(d) == typeof(cd)
        Fun(x->x, d) # fall back to constructor, can't use `identity` as that creates a loop
    else
        # this allows support for singularities, that the constructor doesn't
        sf=fromcanonical(d,Fun(identity,cd))
        Fun(setdomain(space(sf),d),coefficients(sf))
    end
end

Fun(::typeof(identity), S::Space) = Fun(identity,domain(S))



Fun(f::typeof(zero), d::Space) = zeros(eltype(domain(d)),d)
Fun(f::typeof(one), d::Space) = ones(eltype(domain(d)),d)

# Fun(f::Type, d::Domain) = Fun(f,Space(d))
"""
    Fun(f, d::Domain)

Return `Fun(f, Space(d))`, that is, it uses the default space for the specified
domain.

# Examples
```jldoctest
julia> f = Fun(x->x^2, 0..1);

julia> f(0.1) ≈ (0.1)^2
true
```
"""
Fun(f, d::Domain) = Fun(f,Space(d))


# this is the main constructor
"""
    Fun(f, s::Space)

Return a `Fun` representing the function, number, or vector `f` in the
space `s`.  If `f` is vector-valued, it Return a vector-valued analogue
of `s`.

# Examples
```jldoctest
julia> f = Fun(x->x^2, Chebyshev())
Fun(Chebyshev(), [0.5, 0.0, 0.5])

julia> f(0.1) == (0.1)^2
true
```
"""
Fun(f, d::Space) = default_Fun(f, d)

# this supports expanding a Fun to a larger or smaller domain.
# we take the union and then intersection to get at any singularities
# TODO: singularities in space(f)
Fun(f::Fun, d::Domain) = Fun(f,Space((d ∪ domain(f)) ∩ d))





## Aliases



Fun(T::Type,n::Integer) = Fun(T(),n)
Fun(f,n::Integer) = Fun(f,ChebyshevInterval(),n)
Fun(T::Type,d::AbstractVector) = Fun(T(),d)

Fun(f::Fun{SequenceSpace},s::Space) = Fun(s,f.coefficients)

"""
    Fun(f)

Return `Fun(f, space)` by choosing an appropriate `space` for the function.
For univariate functions, `space` is chosen to be `Chebyshev()`, whereas for
multivariate functions, it is a tensor product of `Chebyshev()` spaces.

# Examples
```jldoctest
julia> f = Fun(x -> x^2)
Fun(Chebyshev(), [0.5, 0.0, 0.5])

julia> f(0.1) == (0.1)^2
true

julia> f = Fun((x,y) -> x + y);

julia> f(0.1, 0.2) ≈ 0.3
true
```
"""
function Fun(f::Function)
    if hasonearg(f)
        # check for tuple
        try
            f(0)
        catch ex
            if ex isa BoundsError
                # assume its a tuple
                return Fun(f,ChebyshevInterval()^2)
            else
                rethrow()
            end
        end

        Fun(f,ChebyshevInterval())
    elseif hasnumargs(f,2)
            Fun(f,ChebyshevInterval()^2)
    else
        error("Function not defined on interval or square")
    end
end

