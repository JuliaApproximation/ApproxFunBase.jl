abstract type MultivariateFun{T,N} <: Function end
const BivariateFun{T} = MultivariateFun{T,2}

export grad, lap, curl

#implements coefficients/values/evaluate
space(f::MultivariateFun{T,N}) where {T,N}=mapreduce(k->space(f,k),⊗,1:N)
domain(f::MultivariateFun{T,N}) where {T,N}=mapreduce(k->domain(f,k),×,1:N)

domain(f::MultivariateFun,k::Integer)=domain(space(f,k))

differentiate(u::BivariateFun,i::Integer,j::Integer) =
    j==0 ? u : differentiate(differentiate(u,i),i,j-1)
grad(u::BivariateFun) = [differentiate(u,1),differentiate(u,2)]
lap(u::BivariateFun) = differentiate(u,1,2)+differentiate(u,2,2)
Base.div(u::AbstractVector{<:BivariateFun}) =
    differentiate(u[1],1)+differentiate(u[2],2)
curl(u::AbstractVector{B}) where {B<:BivariateFun} = differentiate(u[2],1)-differentiate(u[1],2)

∇(F::MultivariateFun) = grad(F)
LinearAlgebra.dot(::typeof(∇), F::Vector{<:MultivariateFun}) = div(F)
LinearAlgebra.cross(::typeof(∇), F::Vector{<:MultivariateFun}) = curl(F)

Base.chop(f::MultivariateFun) = chop(f,10eps())
cfstype(::MultivariateFun{T}) where {T} = T
cfstype(::Type{MultivariateFun{T,N}}) where {T,N} = T
cfstype(::Type{MF}) where {MF<:MultivariateFun} = cfstype(supertype(MF))

include("VectorFun.jl")
include("TensorSpace.jl")
include("LowRankFun.jl")
include("ProductFun.jl")
include("TrivialTensorFun.jl")


arglength(f)=length(Base.uncompressed_ast(f.code.def).args[1])



## Convert between Fun and MultivariateFun
# need to chop extra zeros
Fun(f::ProductFun) =
    Fun(space(f),chop!(fromtensor(space(f),coefficients(f)),0))
Fun(f::ProductFun,sp::TensorSpace) = Fun(ProductFun(f,sp))
Fun(f::LowRankFun) = Fun(ProductFun(f))


Fun(f::MultivariateFun,sp::Space) = Fun(Fun(f),sp)

Fun(f,d1::Domain,d2::Domain) = Fun(f,d1*d2)

coefficients(f::BivariateFun,sp::TensorSpace)=coefficients(f, factors(sp)...)



points(f::BivariateFun,k...)=points(space(f),size(f,1),size(f,2),k...)


function *(vx::LowRankFun, u0::ProductFun)
    sum(zip(vx.A, vx.B)) do (a,b)
        transpose(b*(transpose(a*u0)))
    end
end

*(a::ProductFun,b::LowRankFun)=b*a
*(a::MultivariateFun,b::MultivariateFun)=LowRankFun(a)*ProductFun(b)

@inline function ^(a::MultivariateFun, n::Integer)
    n < 0 && return ^(inv(a), -n)
    n == 0 && return one(a)
    n == 1 && return a
    n == 2 && return a * a
    return foldr(*, fill(a, n-2), init=a*a)
end

for OP in (:+,:-,:*,:/)
    @eval begin
        $OP(f::Fun,g::MultivariateFun)=$OP(ProductFun(f),g)
        $OP(f::MultivariateFun,g::Fun)=$OP(f,ProductFun(g))
    end
end


Base.sum(f::Fun{TS},k::Integer) where {TS<:TensorSpace}=sum(ProductFun(f),k)
Base.sum(f::Fun{TS}) where {TS<:TensorSpace}=sum(ProductFun(f))


## kron
# TODO: generalize
function Base.kron(f::Fun,g::Fun)
    sp=space(f)⊗space(g)
    it=tensorizer(sp)
    N=ncoefficients(f);M=ncoefficients(g)
    cfs=Array{promote_type(cfstype(f),cfstype(g))}(undef,0)
    for (k,j) in it
        # Tensor product is N x M, so if we are outside
        # the (N+M)th diagonal we have no more entries
        if k+j > N+M
            break
        elseif k ≤ N && j ≤ M
            push!(cfs,f.coefficients[k]*g.coefficients[j])
        else
            push!(cfs,0)
        end
    end
    Fun(sp,cfs)
end
Base.kron(f::Fun,g::Number) = kron(f,Fun(g))
Base.kron(f::Number,g::Fun) = kron(Fun(f),g)
