import Base: chop

# BLAS/linear algebra overrides

@inline dot(x...) = LinearAlgebra.dot(x...)
@inline dot(M::Int,a::Ptr{T},incx::Int,b::Ptr{T},incy::Int) where {T<:Union{Float64,Float32}} =
    BLAS.dot(M,a,incx,b,incy)
@inline dot(M::Int,a::Ptr{T},incx::Int,b::Ptr{T},incy::Int) where {T<:Union{ComplexF64,ComplexF32}} =
    BLAS.dotc(M,a,incx,b,incy)

dotu(f::StridedVector{T}, g::StridedVector{T}) where {T<:Union{ComplexF32,ComplexF64}} =
    BLAS.dotu(f,g)
dotu(f::AbstractVector{<:Complex}, g::AbstractVector{<:Real}) = dot(conj(f),g)
dotu(f::AbstractVector{<:Real}, g::AbstractVector{<:Real}) = dot(f,g)
function dotu(f::AbstractVector{<:Number}, g::AbstractVector{<:Number})
    Base.require_one_based_indexing(f)
    axes(f) == axes(g) || throw(ArgumentError("vectors must have the same indices"))
    mapreduce(*, +, f, g)
end


normalize!(w::AbstractVector) = rmul!(w,inv(norm(w)))
normalize!(w::Vector{T}) where {T<:BlasFloat} = normalize!(length(w),w)
normalize!(n,w::Union{Vector{T},Ptr{T}}) where {T<:Union{Float64,Float32}} =
    BLAS.scal!(n,inv(BLAS.nrm2(n,w,1)),w,1)
normalize!(n,w::Union{Vector{T},Ptr{T}}) where {T<:Union{ComplexF64,ComplexF32}} =
    BLAS.scal!(n,T(inv(BLAS.nrm2(n,w,1))),w,1)


flipsign(x,y) = Base.flipsign(x,y)
flipsign(x,y::Complex) = x * (z = sign(y); iszero(y) ? one(z) : z)

# Used for spaces not defined yet
struct UnsetNumber <: Number  end
promote_rule(::Type{UnsetNumber},::Type{N}) where {N<:Number} = N
promote_rule(::Type{Bool}, ::Type{UnsetNumber}) = Bool

# Test the number of arguments a function takes
hasnumargs(f,k) = k == 1 ? applicable(f, 0.0) : applicable(f, (1.0:k)...)
hasonearg(f) = hasnumargs(f, 1)

# fast implementation of isapprox with atol a non-keyword argument in most cases
isapprox_atol(a,b,atol;kwds...) = isapprox(a,b;atol=atol,kwds...)
isapprox_atol(a::SVector,b::SVector,atol::Real=0;kwds...) = isapprox_atol(collect(a),collect(b),atol;kwds...)
function isapprox_atol(x::Number, y::Number, atol::Real=0; rtol::Real=Base.rtoldefault(x,y))
    x == y || (isfinite(x) && isfinite(y) && abs(x-y) <= atol + rtol*max(abs(x), abs(y)))
end
function isapprox_atol(x::AbstractArray{T}, y::AbstractArray{S},atol::Real=0; rtol::Real=Base.rtoldefault(T,S), norm::Function=vecnorm) where {T<:Number,S<:Number}
    d = norm(x - y)
    if isfinite(d)
        return d <= atol + rtol*max(norm(x), norm(y))
    else
        # Fall back to a component-wise approximate comparison
        return all(ab -> isapprox(ab[1], ab[2]; rtol=rtol, atol=atol), zip(x, y))
    end
end

# The second case handles zero
isapproxinteger(::Integer) = true
isapproxinteger(x) = isinteger(x) || isapprox(x,round(Int,x))  || isapprox(x+1,round(Int,x+1))


real(x::UnsetNumber) = x
real(::Type{UnsetNumber}) = UnsetNumber

float(x::UnsetNumber) = x
float(::Type{UnsetNumber}) = UnsetNumber

# This creates ApproxFunBase.eps, which we override for default julia types
eps(x...) = Base.eps(x...)
eps(x) = Base.eps(x)

eps(::Type{T}) where T<:Integer = zero(T)
eps(::Type{T}) where T<:Rational = zero(T)
eps(::T) where T<:Integer = eps(T)

eps(::Type{Complex{T}}) where {T<:Real} = eps(real(T))
eps(z::Complex{T}) where {T<:Real} = eps(abs(z))
eps(::Type{Dual{Complex{T}}}) where {T<:Real} = eps(real(T))
eps(z::Dual{Complex{T}}) where {T<:Real} = eps(abs(z))


eps(::Type{Vector{T}}) where {T<:Number} = eps(T)
eps(::Type{SVector{k,T}}) where {k,T<:Number} = eps(T)


isnan(x) = Base.isnan(x)
isnan(x::SVector) = map(isnan,x)


# BLAS


# implement muladd default
muladd(a,b,c) = a*b+c
muladd(a::Number,b::Number,c::Number) = Base.muladd(a,b,c)


for TYP in (:Float64,:Float32,:ComplexF64,:ComplexF32)
    @eval scal!(n::Integer,cst::$TYP,ret::DenseArray{T},k::Integer) where {T<:$TYP} =
            BLAS.scal!(n,cst,ret,k)
end


scal!(n::Integer,cst::BlasFloat,ret::DenseArray{T},k::Integer) where {T<:BlasFloat} =
    BLAS.scal!(n,strictconvert(T,cst),ret,k)

@inline function scal!(n::Integer,cst::Number,ret::AbstractArray,k::Integer)
    @boundscheck checkbounds(ret, 1:(k*(n-1)+1))
    @simd for j=1:k:k*(n-1)+1
        @inbounds ret[j] *= cst
    end
    ret
end

scal!(cst::Number,v::AbstractArray) = scal!(length(v),cst,v,1)



# Helper routines

function reverseeven!(x::AbstractVector)
    Base.require_one_based_indexing(x)
    n = length(x)
    if iseven(n)
        @inbounds @simd for k=2:2:n÷2
            x[k],x[n+2-k] = x[n+2-k],x[k]
        end
    else
        @inbounds @simd for k=2:2:n÷2
            x[k],x[n+1-k] = x[n+1-k],x[k]
        end
    end
    x
end

function negateeven!(x::AbstractVector)
    Base.require_one_based_indexing(x)
    v = view(x, 2:2:length(x))
    v .*= -1
    x
end

#checkerboard, same as applying negativeeven! to all rows then all columns
function negateeven!(X::AbstractMatrix)
    Base.require_one_based_indexing(X)
    for j = 1:2:size(X,2)
        @inbounds @simd for k = 2:2:size(X,1)
            X[k,j] *= -1
        end
    end
    for j = 2:2:size(X,2)
        @inbounds @simd for k = 1:2:size(X,1)
            X[k,j] *= -1
        end
    end
    X
end

const alternatesign! = negateeven!

alternatesign(v::AbstractVector) = alternatesign!(copy(v))

function alternatingsum(v::AbstractVector)
    sum(((a,b),) -> a*b, zip(v, Iterators.cycle((1,-1))))
end

# Sum Hadamard product of vectors up to minimum over lengths
function mindotu(a::AbstractVector,b::AbstractVector)
    Base.require_one_based_indexing(a)
    Base.require_one_based_indexing(b)
    m = min(length(a), length(b))
    dotu(view(a, 1:m), view(b, 1:m))
end


# efficiently resize a Matrix.  Note it doesn't change the input ptr
function unsafe_resize!(W::AbstractMatrix,::Colon,m::Integer)
    if m == size(W,2)
        W
    else
        n=size(W,1)
        reshape(resize!(vec(W),n*m),n,m)
    end
end

function unsafe_resize!(W::AbstractMatrix,n::Integer,::Colon)
    N=size(W,1)
    if n == N
        W
    elseif n < N
        W[1:n,:]
    else
        m=size(W,2)
        ret=Matrix{eltype(W)}(undef, n,m)
        ret[1:N,:] = W
        ret
    end
end

function unsafe_resize!(W::AbstractMatrix,n::Integer,m::Integer)
    N=size(W,1)
    if n == N
        unsafe_resize!(W,:,m)
    else
        unsafe_resize!(unsafe_resize!(W,n,:),:,m)
    end
end


function pad!(f::AbstractVector, n::Integer)
    m = length(f)
	resize!(f,n)
	if n > m
        z = m > 0 ? zero(f[1]) : zero(eltype(f))
        for i in m+1:n
            f[i] = z
        end
	end
    f
end

pad(f::AbstractVector, n::Integer) = pad!(Vector(f), n)

function pad(f::AbstractVector{Any},n::Integer)
	if n > length(f)
        Any[f; zeros(n - length(f))]
	else
        f[1:n]
	end
end

pad(x::Number, n::Int) = [x; zeros(typeof(x), n-1)]

function pad(v::AbstractVector,n::Integer,m::Integer)
    @assert m==1
    pad(v,n)
end

function pad(A::AbstractMatrix,n::Integer,m::Integer)
    Base.require_one_based_indexing(A)
    T=eltype(A)
	if n <= size(A,1) && m <= size(A,2)
        strictconvert(Matrix{T}, A[1:n,1:m])
    else
        ret = Matrix{T}(undef,n,m)
        minn=min(n,size(A,1))
        minm=min(m,size(A,2))

        cinds = CartesianIndices((1:minn, 1:minm))
        copyto!(ret, cinds, A, cinds)

        cinds = CartesianIndices((minn+1:n, 1:minm))
        ret[cinds] .= zero(T)

        cinds = CartesianIndices((axes(ret,1), minm+1:m))
        ret[cinds] .= zero(T)

        ret
	end
end

pad(A::AbstractMatrix,::Colon,m::Integer) = pad(A,size(A,1),m)
pad(A::AbstractMatrix,n::Integer,::Colon) = pad(A,n,size(A,2))


function pad(v, ::PosInfinity)
    if isinf(length(v))
        v
    else
        Vcat(v, Zeros{Int}(∞))
    end
end

function pad(v::AbstractVector{T}, ::PosInfinity) where T
    if isinf(length(v))
        v
    else
        Vcat(v, Zeros{T}(∞))
    end
end

_pad!!(::Val{false}) = pad
_pad!!(::Val{true}) = pad!

#TODO:padleft!

function padleft(f::AbstractVector,n::Integer)
	if (n > length(f))
        [zeros(n - length(f)); f]
	else
        f[end-n+1:end]
	end
end



##chop!
function chop!(c::AbstractVector,tol::Real)
    @assert tol >= 0

    for k=length(c):-1:1
        if abs(c[k]) > tol
            resize!(c,k)
            return c
        end
    end

    resize!(c,0)
    c
end

chop(f::AbstractVector,tol) = chop!(copy(f),tol)
chop!(f::AbstractVector) = chop!(f,eps())


function chop!(A::AbstractArray,tol)
    for k=size(A,1):-1:1
        if norm(A[k,:])>tol
            A=A[1:k,:]
            break
        end
    end
    for k=size(A,2):-1:1
        if norm(A[:,k])>tol
            A=A[:,1:k]
            break
        end
    end
    return A
end
chop(A::AbstractArray,tol)=chop!(A,tol)#replace by chop!(copy(A),tol) when chop! is actually in-place.



## interlace



function interlace(v::Union{Vector{Any},Tuple})
    #determine type
    T=Float64
    for vk in v
        if isa(vk,Vector{Complex{Float64}})
            T=Complex{Float64}
        end
    end
    b=Vector{Vector{T}}(undef, length(v))
    for k=1:length(v)
        b[k]=v[k]
    end
    interlace(b)
end

initvector(::Type{T}, n) where {T<:Number} = zeros(T, n)
initvector(::Type{T}, n) where {T} = Vector{T}(undef, n)

function interlace(a::AbstractVector, b::AbstractVector, (ncomponents_a, ncomponents_b) = (1,1))
    T=promote_type(eltype(a), eltype(b))

    # we pad the arrays first to ensure that the leading blocks are full,
    # that is if the space corresponding to a is (S1 ⊕ S2) and that for b is S3,
    # and a = [1,2,3] and b = [5,6,7,8], the resulting coefficients would be
    # [1,2, 5, 3,0, 6, 0,0, 7, 0,0, 8], so a is padded to [1,2,3,0,0,0,0,0]
    # Second example: if a = [1,2,3] and b = [5,6], the result would be
    # [1,2, 5, 3,0, 6], so a is padded to [1,2,3,0]
    # Third example: if a = [1,2,3] and b = [5], the result would be
    # [1,2, 5, 3]. In this case there is no padding in either a or b
    # Fourth example: if a = [1,2,3,4,11,12] and b = [5], the result would be
    # [1,2, 5, 3,4, 0, 11,12]. In this case, b is padded to [5,0]

    nblk_a = cld(length(a), ncomponents_a)
    nblk_b = cld(length(b), ncomponents_b)

    if nblk_b >= nblk_a
        nblk_a = nblk_b
    elseif nblk_a - 1 > nblk_b
        nblk_b = nblk_a-1
    end

    pad_a = pad(a, ncomponents_a * nblk_a)
    pad_b = pad(b, ncomponents_b * nblk_b)

    blksz_a = Fill(ncomponents_a, nblk_a)
    aPBlk = PseudoBlockArray(pad_a, blksz_a)
    blksz_b = Fill(ncomponents_b, nblk_b)
    bPBkl = PseudoBlockArray(pad_b, blksz_b)

    nblk_ret = nblk_a + nblk_b
    blksz_ret = zeros(Int, nblk_ret)
    blksz_ret[1:2:end] = blksz_a
    blksz_ret[2:2:end] = blksz_b
    nret = sum(blksz_ret)
    ret = initvector(T, nret)
    retPBlk = PseudoBlockArray(ret, blksz_ret)

    @views begin
        for (ind, i) in enumerate(1:2:nblk_ret)
            retPBlk[Block(i)] = aPBlk[Block(ind)]
        end
        for (ind, i) in enumerate(2:2:nblk_ret)
            retPBlk[Block(i)] = bPBkl[Block(ind)]
        end
    end
    resize!(ret, findlast(!iszero, ret))
end

### In-place O(n) interlacing

function highestleader(n::Int)
    i = 1
    while 3i < n i *= 3 end
    i
end

function nextindex(i::Int,n::Int)
    i <<= 1
    while i > n
        i -= n + 1
    end
    i
end

function cycle_rotate!(v::AbstractVector, leader::Int, it::Int, twom::Int)
    i = nextindex(leader, twom)
    while i != leader
        idx1, idx2 = it + i - 1, it + leader - 1
        @inbounds v[idx1], v[idx2] = v[idx2], v[idx1]
        i = nextindex(i, twom)
    end
    v
end

function right_cyclic_shift!(v::AbstractVector, it::Int, m::Int, n::Int)
    itpm = it + m
    itpmm1 = itpm - 1
    itpmpnm1 = itpmm1 + n
    reverse!(v, itpm, itpmpnm1)
    reverse!(v, itpm, itpmm1 + m)
    reverse!(v, itpm + m, itpmpnm1)
    v
end

"""
This function implements the algorithm described in:

    P. Jain, "A simple in-place algorithm for in-shuffle," arXiv:0805.1598, 2008.
"""
function interlace!(v::AbstractVector,offset::Int)
    N = length(v)
    if N < 2 + offset
        return v
    end

    it = 1 + offset
    m = 0
    n = 1

    while m < n
        twom = N + 1 - it
        h = highestleader(twom)
        m = h > 1 ? h÷2 : 1
        n = twom÷2

        right_cyclic_shift!(v,it,m,n)

        leader = 1
        while leader < 2m
            cycle_rotate!(v, leader, it, 2m)
            leader *= 3
        end

        it += 2m
    end
    v
end

## slnorm gives the norm of a slice of a matrix

function slnorm(u::AbstractMatrix,r::AbstractRange,::Colon)
    ret = zero(real(eltype(u)))
    for k=r
        @simd for j=1:size(u,2)
            #@inbounds
            ret=max(norm(u[k,j]),ret)
        end
    end
    ret
end


function slnorm(m::AbstractMatrix,kr::AbstractRange,jr::AbstractRange)
    ret=zero(real(eltype(m)))
    for j=jr
        nrm=zero(real(eltype(m)))
        for k=kr
            @inbounds nrm+=abs2(m[k,j])
        end
        ret=max(sqrt(nrm),ret)
    end
    ret
end

slnorm(m::AbstractMatrix,kr::AbstractRange,jr::Integer) = slnorm(m,kr,jr:jr)
slnorm(m::AbstractMatrix,kr::Integer,jr::AbstractRange) = slnorm(m,kr:kr,jr)


function slnorm(B::BandedMatrix{T},r::AbstractRange,::Colon) where T
    ret = zero(real(T))
    m=size(B,2)
    for k=r
        @simd for j=max(1,k-B.l):min(k+B.u,m)
            #@inbounds
            ret=max(norm(B[k,j]),ret)
        end
    end
    ret
end


slnorm(m::AbstractMatrix,k::Integer,::Colon) = slnorm(m,k,1:size(m,2))
slnorm(m::AbstractMatrix,::Colon,j::Integer) = slnorm(m,1:size(m,1),j)


## Infinity



Base.isless(x::Block{1}, y::PosInfinity) = isless(Int(x), y)
Base.isless(x::PosInfinity, y::Block{1}) = isless(x, Int(y))


## BandedMatrix



pad!(A::BandedMatrix,n,::Colon) = pad!(A,n,n+A.u)  # Default is to get all columns
columnrange(A,row::Integer) = max(1,row-bandwidth(A,1)):row+bandwidth(A,2)



## Store iterator
mutable struct CachedIterator{T,IT}
    iterator::IT
    storage::Vector{T}
    state
    length::Int
end

CachedIterator{T,IT}(it::IT, state) where {T,IT} = CachedIterator{T,IT}(it,T[],state,0)
CachedIterator(it::IT) where IT = CachedIterator{eltype(it),IT}(it, ())

function Base.show(io::IO, c::CachedIterator)
    print(io, "Cached ", c.iterator, " with ", c.length, " stored elements, and state = ", c.state)
end

function resize!(it::CachedIterator{T},n::Integer) where {T}
    m = it.length
    if n > m
        if n > length(it.storage)
            resize!(it.storage,2n)
        end

        @inbounds for k = m+1:n
            xst = iterate(it.iterator,it.state...)
            if xst === nothing
                it.length = k-1
                return it
            end
            v::T, st = xst
            it.storage[k] = v
            it.state = (st,)
        end
        it.length = n
    end
    it
end


eltype(it::Type{<:CachedIterator{T}}) where {T} = T

function Base.IteratorSize(::Type{<:CachedIterator{<:Any,IT}}) where {IT}
    Base.IteratorSize(IT) isa Base.IsInfinite ? Base.IsInfinite() : Base.HasLength()
end

Base.keys(c::CachedIterator) = oneto(length(c))

iterate(it::CachedIterator) = iterate(it,1)
function iterate(it::CachedIterator,st::Int)
    if  st > it.length && iterate(it.iterator,it.state...) === nothing
        nothing
    else
        (it[st],st+1)
    end
end

function getindex(it::CachedIterator, k)
    mx = maximum(k)
    if mx > length(it) || mx < 1
        throw(BoundsError(it,k))
    end
    v = resize!(it, mx)
    v.storage[k]
end

@deprecate findfirst(A::CachedIterator, x) findfirst(x, A::CachedIterator)
findfirst(x::T, A::CachedIterator{T}) where {T} = findfirst(==(x), A)

length(A::CachedIterator) = length(A.iterator)

## nocat
vnocat(A...) = Base.vect(A...)
hnocat(A...) = Base.typed_hcat(mapreduce(typeof,promote_type,A),A...)
hvnocat(rows,A...) = Base.typed_hvcat(mapreduce(typeof,promote_type,A),rows,A...)
macro nocat(ex)
    head = ex.head
    ex.head = :call
    if head == :vcat
        fn = Expr(:., :ApproxFunBase, QuoteNode(:vnocat))
        insert!(ex.args, 1, fn)
    elseif head == :call && ex.args[1] == :vcat
        fn = Expr(:., :ApproxFunBase, QuoteNode(:vnocat))
        ex.args[1] = fn
    elseif head == :hcat
        fn = Expr(:., :ApproxFunBase, QuoteNode(:hnocat))
        insert!(ex.args, 1, fn)
    elseif head == :call && ex.args[1] == :hcat
        fn = Expr(:., :ApproxFunBase, QuoteNode(:hnocat))
        ex.args[1] = fn
    elseif head == :hvcat
        fn = Expr(:., :ApproxFunBase, QuoteNode(:hvnocat))
        insert!(ex.args, 1, fn)
    elseif head == :call && ex.args[1] == :hvcat
        fn = Expr(:., :ApproxFunBase, QuoteNode(:hvnocat))
        ex.args[1] = fn
    else
        throw(ArgumentError("@nocat can only be used with vcat/hcat/hvcat expressions"))
    end
    esc(ex)
end



# TODO: deprecate
dynamic(f) = f

# Matrix inputs




## conv

conv(x::AbstractVector, y::AbstractVector) = DSP.conv(x, y)
@generated function conv(x::SVector{N}, y::SVector{M}) where {N,M}
    NM = N+M-1
    quote
        strictconvert(SVector{$NM}, DSP.conv(Vector(x), Vector(y)))
    end
end

conv(x::SVector{1}, y::SVector{1}) = x.*y
conv(x::AbstractVector, y::SVector{1}) = x*y[1]
conv(y::SVector{1}, x::AbstractVector) = y[1]*x
conv(x::AbstractFill, y::SVector{1}) = x*y[1]
conv(y::SVector{1}, x::AbstractFill) = y[1]*x
conv(x::AbstractFill, y::AbstractFill) = DSP.conv(x, y)


## BlockInterlacer
# interlaces coefficients by blocks
# this has the property that all the coefficients of a block of a subspace
# are grouped together, starting with the first bloc
#
# TODO: cache sums


struct BlockInterlacer{DMS<:Tuple{Vararg{AbstractVector{Int}}}}
    blocks::DMS
end


const TrivialInterlacer{d} = BlockInterlacer{<:NTuple{d,Ones{Int}}}

BlockInterlacer(v::AbstractVector) = BlockInterlacer(Tuple(v))

eltype(::Type{<:BlockInterlacer}) = Tuple{Int,Int}

dimensions(b::BlockInterlacer) = map(sum,b.blocks)
dimension(b::BlockInterlacer,k) = sum(b.blocks[k])
length(b::BlockInterlacer) = mapreduce(sum,+,b.blocks)

Base.IteratorSize(::Type{BlockInterlacer{T}}) where {T} = _IteratorSize(T)

Base.show(io::IO, B::BlockInterlacer) = print(io, BlockInterlacer, "(", B.blocks, ")")

# the state is always (whichblock,curblock,cursubblock,curcoefficients)
# start(it::BlockInterlacer) = (1,1,map(start,it.blocks),ntuple(zero,length(it.blocks)))



# are all Ints, so finite dimensional
function done(it::BlockInterlacer,st)
    lngs = st[end]
    for (k, (itk, lk)) in enumerate(zip(it.blocks, lngs))
        if lk < sum(itk)
            return false
        end
    end
    return true
end

iterate(it::BlockInterlacer) =
    iterate(it, (1,1,ntuple(_ -> tuple(), length(it.blocks)),
            ntuple(zero,length(it.blocks))))

function iterate(it::BlockInterlacer, (N,k,blkst,lngs))
    done(it, (N,k,blkst,lngs)) && return nothing

    if N > length(it.blocks)
        # increment to next block
        blkst = map(it.blocks,blkst) do blit,blst
                xblst = iterate(blit, blst...)
                xblst === nothing ? blst : (xblst[2],)
            end
        return iterate(it,(1,1,blkst,lngs))
    end

    Bnxtb = iterate(it.blocks[N],blkst[N]...)  # B is block size

    if Bnxtb === nothing
        # increment to next N
        return iterate(it,(N+1,1,blkst,lngs))
    end

    B,nxtb = Bnxtb

    if k > B
        #increment to next N
        return iterate(it,(N+1,1,blkst,lngs))
    end


    lngs = Base.setindex(lngs, lngs[N]+1, N)
    return (N,lngs[N]),(N,k+1,blkst,lngs)
end

cache(Q::BlockInterlacer) = CachedIterator(Q)
