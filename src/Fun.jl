export Fun, evaluate, values, points, extrapolate, setdomain
export coefficients, ncoefficients, coefficient
export integrate, differentiate, domain, space, linesum, linenorm

##  Constructors


mutable struct Fun{S,T,VT} <: Function
    space::S
    coefficients::VT
    function Fun{S,T,VT}(sp::S, coeff::VT) where {S,T,VT}
        axes(sp,2) == axes(coeff,1) || throw(DimensionMismatch(""))
        new{S,T,VT}(sp,coeff)
    end
end

const VFun{S,T} = Fun{S,T,Vector{T}}

_pad(c, _, n) = [c; zeros(eltype(c), n-length(c))]
_pad(c, ::Infinity, ::Infinity) = c
pad(c, n::Integer) = _pad(c, length(c), n)
function Fun(sp::Space, c::AbstractVector) 
    coeff = pad(c, size(sp,2))
    Fun{typeof(sp),eltype(coeff),typeof(coeff)}(sp, coeff)
end
Fun() = Fun(identity)
Fun(d) = Fun(identity, d)

Fun(v::AbstractQuasiVector) = Fun(arguments(v)...)
function Fun(f::Function, S::Space)
    if !applicable(f, checkpoints(S)[1])
        Fun(x -> f(x...), S)
    else
        Fun(S, S \ f.(axes(S,1)))
    end
end

Fun(f::Function, d) = Fun(f, Space(d))


function Fun(sp::Space, v::AbstractVector{Any})
    if isempty(v)
        Fun(sp, Float64[])
    elseif all(x->isa(x,Number),v)
        Fun(sp, Vector{mapreduce(typeof,promote_type,v)}(v))
    else
        error("Cannot construct Fun with coefficients $v and space $sp")
    end
end


hasnumargs(f::Fun, k) = k == 1 || domaindimension(f) == k  # all funs take a single argument as a Vec

##Coefficient routines
#TODO: domainscompatible?


coefficients(f::Fun) = f.coefficients

vec(f::Fun) = f.space * coefficients(f)
coefficients(f::Fun, msp::Space) = msp \ vec(f)
coefficients(c::Number,sp::Space) = coefficients(Fun(c,sp))
Fun(f::Fun, S::Space) = Fun(S, coefficients(f, S))

##Convert routines


convert(::Type{Fun{S,T,VT}}, f::Fun{S}) where {T,S,VT} =
    Fun(f.space,convert(VT,f.coefficients))
convert(::Type{Fun{S,T,VT}}, f::Fun) where {T,S,VT} =
    Fun(Fun(f.space,convert(VT,f.coefficients)),convert(S,space(f)))

convert(::Type{Fun{S,T}}, f::Fun{S}) where {T,S} =
    Fun(f.space,convert(AbstractVector{T},f.coefficients))


convert(::Type{VFun{S,T}}, x::Number) where {T,S} =
    x==0 ? zeros(T,S(AnyDomain())) : x*ones(T,S(AnyDomain()))
convert(::Type{Fun{S}}, x::Number) where {S} =
    x==0 ? zeros(S(AnyDomain())) : x*ones(S(AnyDomain()))
convert(::Type{IF}, x::Number) where {IF<:Fun} = convert(IF,Fun(x))

Fun{S,T,VT}(f::Fun) where {S,T,VT} = convert(Fun{S,T,VT}, f)
Fun{S,T}(f::Fun) where {S,T} = convert(Fun{S,T}, f)
Fun{S}(f::Fun) where {S} = convert(Fun{S}, f)

# if we are promoting, we need to change to a VFun
Base.promote_rule(::Type{Fun{S,T,VT1}},::Type{Fun{S,V,VT2}}) where {T,V,S,VT1,VT2} =
    VFun{S,promote_type(T,V)}


# TODO: Never assume!
Base.promote_op(::typeof(*), ::Type{F1}, ::Type{F2}) where {F1<:Fun,F2<:Fun} =
    promote_type(F1,F2) # assume multiplication is defined between same types

# we know multiplication by numbers preserves types
Base.promote_op(::typeof(*),::Type{N},::Type{Fun{S,T,VT}}) where {N<:Number,S,T,VT} =
    VFun{S,promote_type(T,N)}
Base.promote_op(::typeof(*),::Type{Fun{S,T,VT}},::Type{N}) where {N<:Number,S,T,VT} =
    VFun{S,promote_type(T,N)}

Base.promote_op(::typeof(LinearAlgebra.matprod),::Type{Fun{S1,T1,VT1}},::Type{Fun{S2,T2,VT2}}) where {S1,T1,VT1,S2,T2,VT2} =
            VFun{promote_type(S1,S2),promote_type(T1,T2)}
# Fun's are always vector spaces, so we know matprod will preserve the space
Base.promote_op(::typeof(LinearAlgebra.matprod),::Type{Fun{S,T,VT}},::Type{NN}) where {S,T,VT,NN<:Number} =
            VFun{S,promote_type(T,NN)}
Base.promote_op(::typeof(LinearAlgebra.matprod),::Type{NN},::Type{Fun{S,T,VT}}) where {S,T,VT,NN<:Number} =
            VFun{S,promote_type(T,NN)}



zero(::Type{Fun}) = Fun(0.)
zero(::Type{Fun{S,T,VT}}) where {T,S<:Space,VT} = zeros(T,S(AnyDomain()))
one(::Type{Fun{S,T,VT}}) where {T,S<:Space,VT} = ones(T,S(AnyDomain()))
zero(f::Fun{S,T}) where {S,T} = zeros(T,f.space)
one(f::Fun{S,T}) where {S,T} = ones(T,f.space)

cfstype(::Fun{S,T}) where {S,T} = T
cfstype(::Type{Fun{S,T,VT}}) where {S,T,VT} = T

# Number and Array conform to the Fun interface
cfstype(::Type{T}) where T<: Number = T
cfstype(::T) where T<: Number = T
cfstype(::Type{<:AbstractArray{T}}) where T = T
cfstype(::AbstractArray{T}) where T = T

coefficients(f::Number) = [f]
coefficients(f::AbstractArray) = f


#supports broadcasting and scalar iterator
const ScalarSpace = Space{<:Number}
const RealSpace = Space{<:Real}
const ScalarFun = Fun{<:ScalarSpace}
const ArrayFun = Fun{<:Space{<:AbstractArray}}
const MatrixFun = Fun{<:Space{<:AbstractMatrix}}
const VectorFun = Fun{<:Space{<:AbstractVector}}

size(f::Fun, k...) = size(first(axes(space(f),1)), k...)
length(f::Fun) = length(first(axes(space(f),1)))

getindex(f::ScalarFun, ::CartesianIndex{0}) = f
getindex(f::ScalarFun, k::Integer) = k == 1 ? f : throw(BoundsError())

iterate(x::ScalarFun) = (x, nothing)
iterate(x::ScalarFun, ::Any) = nothing
isempty(x::ScalarFun) = false

iterate(A::ArrayFun, i=1) = (@_inline_meta; (i % UInt) - 1 < length(A) ? (@inbounds A[i], i + 1) : nothing)

in(x::ScalarFun, y::ScalarFun) = x == y

setspace(f::Fun, s::Space) = Fun(s, coefficients(f))


## domain


## General routines


domain(f::Fun) = axes(f.space, 1)
domain(v::AbstractMatrix{T}) where {T<:Fun} = map(domain, v)
domaindimension(f::Fun) = domaindimension(f.space)

setdomain(f::Fun,d::Domain) = Fun(setdomain(space(f),d),f.coefficients)

space(f::Fun) = f.space
spacescompatible(f::Fun, g::Fun) = spacescompatible(space(f),space(g))
pointscompatible(f::Fun, g::Fun) = pointscompatible(space(f),space(g))
canonicalspace(f::Fun) = canonicalspace(space(f))
canonicaldomain(f::Fun) = canonicaldomain(space(f))


##Evaluation

evaluate(f::Fun, x) = vec(f)[x]
evaluate(f::Fun, x, y, z...) = evaluate(f, Vec(x,y,z...))
(f::Fun)(x...) = evaluate(f, x...)

dynamic(f::Fun) = f # Fun's are already dynamic in that they compile by type


## Extrapolation


# Default extrapolation is evaluation. Override this function for extrapolation enabled spaces.
extrapolate(f::AbstractVector, S::Space, x...) = evaluate(f, S, x...)

# Do not override these
extrapolate(f::Fun,x) = extrapolate(f.coefficients,f.space,x)
extrapolate(f::Fun,x,y,z...) = extrapolate(f.coefficients,f.space,Vec(x,y,z...))


##Data routines


values(f::Fun,dat...) = itransform(f.space,f.coefficients,dat...)
points(f::Fun) = points(f.space,ncoefficients(f))
ncoefficients(f::Fun) = nzeros(f.coefficients)

## Manipulate length


function chop!(sp::ScalarSpace, cfs, tol::Real)
    n = standardchoplength(cfs, tol)
    resize!(cfs,n)
    cfs
end

chop!(sp::Space, cfs, tol::Real) = chop!(cfs,maximum(abs,cfs)*tol)
chop!(sp::Space, cfs) = chop!(sp,cfs,10eps())

function chop!(f::Fun,tol...)
    chop!(space(f),f.coefficients,tol...)
    f
end

chop(f::Fun,tol) = chop!(Fun(f.space,copy(f.coefficients)),tol)
chop(f::Fun) = chop!(Fun(f.space,copy(f.coefficients)))

copy(f::Fun) = Fun(space(f),copy(f.coefficients))

## Addition and multiplication



for op in (:+,:-)
    @eval begin
        function $op(f::Fun, g::Fun)
            if space(f) == space(g)
                Fun(space(f), ($op)(coefficients(f), coefficients(g)))
            else
                m = broadcastspace($op, f.space, g.space)
                if m isa NoSpace
                    error("Cannot "*string($op)*" because no space is the union of "*string(typeof(f.space))*" and "*string(typeof(g.space)))
                end
                $op(Fun(f,m), Fun(g,m)) # convert to same space
            end
        end
        $op(f::Fun{S,T},c::T) where {S,T<:Number} = c==0 ? f : $op(f,Fun(c))
        $op(f::Fun,c::Number) = $op(f,Fun(c,space(f)))
        $op(f::Fun,c::UniformScaling) = $op(f,c.λ)
        $op(c::UniformScaling,f::Fun) = $op(c.λ,f)
    end
end


# equivalent to Y+=a*X
axpy!(a, X::Fun, Y::Fun)=axpy!(a,coefficients(X,space(Y)),Y)
function axpy!(a,xcfs::AbstractVector,Y::Fun)
    if a!=0
        n=ncoefficients(Y); m=length(xcfs)

        if n≤m
            resize!(Y.coefficients,m)
            for k=1:n
                @inbounds Y.coefficients[k]+=a*xcfs[k]
            end
            for k=n+1:m
                @inbounds Y.coefficients[k]=a*xcfs[k]
            end
        else #X is smaller
            for k=1:m
                @inbounds Y.coefficients[k]+=a*xcfs[k]
            end
        end
    end

    Y
end



-(f::Fun) = Fun(f.space, -f.coefficients)
-(c::Number,f::Fun) = -(f-c)


for op = (:*,:/)
    @eval $op(f::Fun, c::Number) = Fun(f.space,$op(f.coefficients,c))
end


for op = (:*,:+)
    @eval $op(c::Number, f::Fun) = $op(f,c)
end

\(c::Number, f::Fun) = Fun(f.space, c \ f.coefficients)


function intpow(f::Fun,k::Integer)
    if k == 0
        ones(space(f))
    elseif k==1
        f
    elseif k > 1
        f*f^(k-1)
    else
        1/f^(-k)
    end
end

^(f::Fun, k::Integer) = intpow(f,k)

inv(f::Fun) = 1/f

# Integrals over two Funs, which are fast with the orthogonal weight.


# Having fallbacks allow for the fast implementations.

defaultbilinearform(f::Fun,g::Fun)=sum(f*g)
defaultlinebilinearform(f::Fun,g::Fun)=linesum(f*g)

bilinearform(f::Fun,g::Fun)=defaultbilinearform(f,g)
bilinearform(c::Number,g::Fun)=sum(c*g)
bilinearform(g::Fun,c::Number)=sum(g*c)

linebilinearform(f::Fun,g::Fun)=defaultbilinearform(f,g)
linebilinearform(c::Number,g::Fun)=linesum(c*g)
linebilinearform(g::Fun,c::Number)=linesum(g*c)



# Conjugations

innerproduct(f::Fun,g::Fun)=bilinearform(conj(f),g)
innerproduct(c::Number,g::Fun)=bilinearform(conj(c),g)
innerproduct(g::Fun,c::Number)=bilinearform(conj(g),c)

lineinnerproduct(f::Fun,g::Fun)=linebilinearform(conj(f),g)
lineinnerproduct(c::Number,g::Fun)=linebilinearform(conj(c),g)
lineinnerproduct(g::Fun,c::Number)=linebilinearform(conj(g),c)

## Norm

for (OP,SUM) in ((:(norm),:(sum)),(:linenorm,:linesum))
    @eval begin
        $OP(f::Fun) = $OP(f,2)

        function $OP(f::ScalarFun, p::Real)
            if p < 1
                return error("p should be 1 ≤ p ≤ ∞")
            elseif 1 ≤ p < Inf
                return abs($SUM(abs2(f)^(p/2)))^(1/p)
            else
                return maximum(abs,f)
            end
        end

        function $OP(f::ScalarFun,p::Int)
            if 1 ≤ p < Inf
                return iseven(p) ? abs($SUM(abs2(f)^(p÷2)))^(1/p) : abs($SUM(abs2(f)^(p/2)))^(1/p)
            else
                return error("p should be 1 ≤ p ≤ ∞")
            end
        end
    end
end


## Mapped functions

transpose(f::Fun) = f  # default no-op

for op = (:(real),:(imag),:(conj))
    @eval ($op)(f::Fun{S}) where {S<:RealSpace} = Fun(f.space,($op)(f.coefficients))
end

conj(f::Fun) = error("Override conj for $(typeof(f))")

abs2(f::Fun{S,T}) where {S<:RealSpace,T<:Real} = f^2
abs2(f::Fun{S,T}) where {S<:RealSpace,T<:Complex} = real(f)^2+imag(f)^2
abs2(f::Fun)=f*conj(f)

##  integration

function cumsum(f::Fun)
    cf = integrate(f)
    cf - first(cf)
end

cumsum(f::Fun,d::Domain)=cumsum(Fun(f,d))
cumsum(f::Fun,d)=cumsum(f,Domain(d))



function differentiate(f::Fun,k::Integer)
    @assert k >= 0
    (k==0) ? f : differentiate(differentiate(f),k-1)
end

# use conj(transpose(f)) for ArraySpace
function differentiate(f) 
    v = vec(f)
    D = Derivative(axes(v,1))
    Fun(D*v)
end
adjoint(f::Fun) = differentiate(f)



==(f::Fun,g::Fun) =  (f.coefficients == g.coefficients && f.space == g.space)

coefficientnorm(f::Fun,p::Real=2) = norm(f.coefficients,p)


Base.rtoldefault(::Type{F}) where {F<:Fun} = Base.rtoldefault(cfstype(F))
Base.rtoldefault(x::Union{T,Type{T}}, y::Union{S,Type{S}}, atol) where {T<:Fun,S<:Fun} =
    Base.rtoldefault(cfstype(x),cfstype(y), atol)

Base.rtoldefault(x::Union{T,Type{T}}, y::Union{S,Type{S}}, atol) where {T<:Number,S<:Fun} =
    Base.rtoldefault(cfstype(x),cfstype(y), atol)
Base.rtoldefault(x::Union{T,Type{T}}, y::Union{S,Type{S}}, atol) where {T<:Fun,S<:Number} =
    Base.rtoldefault(cfstype(x),cfstype(y), atol)


function isapprox(f::Fun{S1,T},g::Fun{S2,S};rtol::Real=Base.rtoldefault(T,S,0), atol::Real=0, norm::Function=coefficientnorm) where {S1,S2,T,S}
    if spacescompatible(f,g)
        d = norm(f - g)
        if isfinite(d)
            return d <= atol + rtol*max(norm(f), norm(g))
        else
            # Fall back to a component-wise approximate comparison
            return false
        end
    else
        sp=union(f.space,g.space)
        if isa(sp,NoSpace)
            false
        else
            isapprox(Fun(f,sp),Fun(g,sp);rtol=rtol,atol=atol,norm=norm)
        end
    end
end

isapprox(f::Fun, g::Number) = f ≈ g*ones(space(f))
isapprox(g::Number, f::Fun) = g*ones(space(f)) ≈ f


isreal(f::Fun{<:RealSpace,<:Real}) = true
isreal(f::Fun) = false

iszero(f::Fun)    = all(iszero,f.coefficients)



# sum, integrate, and idfferentiate are in CalculusOperator


function reverseorientation(f::Fun)
    csp=canonicalspace(f)
    if spacescompatible(csp,space(f))
        error("Implement reverseorientation for $(typeof(f))")
    else
        reverseorientation(Fun(f,csp))
    end
end


## non-vector notation

for op in (:+,:-,:*,:/,:^)
    @eval begin
        broadcast(::typeof($op), a::Fun, b::Fun) = $op(a,b)
        broadcast(::typeof($op), a::Fun, b::Number) = $op(a,b)
        broadcast(::typeof($op), a::Number, b::Fun) = $op(a,b)
    end
end

## broadcasting
# for broadcasting, we support broadcasting over `Fun`s, e.g.
#
#       exp.(f) is equivalent to Fun(x->exp(f(x)),domain(f)),
#       exp.(f .+ g) is equivalent to Fun(x->exp(f(x)+g(x)),domain(f) ∪ domain(g)),
#       exp.(f .+ 2) is equivalent to Fun(x->exp(f(x)+2),domain(f)),
#
# When we are broadcasting over arrays and scalar Fun's together,
# it broadcasts over the Array and treats the scalar Fun's as constants, so will not
# necessarily call the constructor:
#
#       exp.( x .+ [1,2,3]) is equivalent to [exp(x + 1),exp(x+2),exp(x+3)]
#
# When broadcasting over Fun's with array values, it treats them like Fun's:
#
#   exp.( [x;x]) throws an error as it is equivalent to Fun(x->exp([x;x](x)),domain(f))
#
# This is consistent with the deprecation thrown by exp.([[1,2],[3,4]). Note that
#
#   exp.( [x,x]) is equivalent to [exp(x),exp(x)]
#
# does not throw the same error. When array values are mixed with arrays, the Array
# takes presidence:
#
#   exp.([x;x] .+ [x,x]) is equivalent to exp.(Array([x;x]) .+ [x,x])
#
# This presidence is picked by the `promote_containertype` overrides.

struct FunStyle <: BroadcastStyle end

BroadcastStyle(::Type{<:Fun}) = FunStyle()

BroadcastStyle(::FunStyle, ::FunStyle) = FunStyle()
BroadcastStyle(::AbstractArrayStyle{0}, ::FunStyle) = FunStyle()
BroadcastStyle(::FunStyle, ::AbstractArrayStyle{0}) = FunStyle()
BroadcastStyle(A::AbstractArrayStyle, ::FunStyle) = A
BroadcastStyle(::FunStyle, A::AbstractArrayStyle) = A


# Treat Array Fun's like Arrays when broadcasting with an Array
# note this only gets called when containertype returns Array,
# so will not be used when no argument is an Array
Base.broadcast_axes(::Type{Fun}, A) = axes(A)
Base.broadcastable(x::Fun) = x

broadcastdomain(b) = AnyDomain()
broadcastdomain(b::Fun) = domain(b)
broadcastdomain(b::Broadcasted) = mapreduce(broadcastdomain, ∪, b.args)

broadcasteval(f::Function, x) = f(x)
broadcasteval(c, x) = c
broadcasteval(c::Ref, x) = c.x
broadcasteval(b::Broadcasted, x) = b.f(broadcasteval.(b.args, x)...)

# TODO: use generated function to improve the following
function copy(bc::Broadcasted{FunStyle})
    d = broadcastdomain(bc)
    Fun(x -> broadcasteval(bc, x), d)
end

function copyto!(dest::Fun, bc::Broadcasted{FunStyle})
    if broadcastdomain(bc) ≠ domain(dest)
        throw(ArgumentError("Domain of right-hand side incompatible with destination"))
    end
    ret = copy(bc)
    cfs = coefficients(ret,space(dest))
    resize!(dest.coefficients, length(cfs))
    dest.coefficients[:] = cfs
    dest
end
