export Evaluation,ivp,bvp,Dirichlet,Neumann

## Evaluation constructors

abstract type Evaluation{T}<:Operator{T} end

@functional Evaluation
evaluation_point(C::Evaluation) = C.x

@enum Boundary RightEndPoint=1 LeftEndPoint=-1

isleftendpoint(_) = false
isrightendpoint(_) = false
isleftendpoint(::typeof(leftendpoint)) = true
isrightendpoint(::typeof(rightendpoint)) = true
isrightendpoint(x::Boundary) = x == RightEndPoint
isleftendpoint(x::Boundary) = x == LeftEndPoint

# M = leftendpoint/rightendpoint if endpoint
struct ConcreteEvaluation{S,M,OT,T} <: Evaluation{T}
    space::S
    x::M
    order::OT
end

ConcreteEvaluation(sp::Space,x,o::Number) =
    ConcreteEvaluation{typeof(sp),typeof(x),typeof(o),rangetype(sp)}(sp,x,o)


Evaluation(::Type{T},sp::Space,x,order) where {T} =
    ConcreteEvaluation{typeof(sp),typeof(x),typeof(order),T}(sp,x,order)
# TODO: This seems like a bad idea: if you are specifying x, just go with the generic version
function Evaluation(::Type{T},sp::UnivariateSpace,x::Number,order) where {T}
    d=domain(sp)
    if isa(d,IntervalOrSegment) && isapprox(leftendpoint(d),x)
        Evaluation(T,sp,leftendpoint,order)
    elseif isa(d,IntervalOrSegment) && isapprox(rightendpoint(d),x)
        Evaluation(T,sp,rightendpoint,order)
    else
        ConcreteEvaluation{typeof(sp),typeof(x),typeof(order),T}(sp,x,order)
    end
end

Evaluation(sp::Space,x,order) = Evaluation(rangetype(sp),sp,x,order)

const SpecialEvalPtType = Union{typeof(leftendpoint),typeof(rightendpoint),Boundary}
const EvalPtType = Union{Number,SpecialEvalPtType}

error_space(d::Space) = error("Override Evaluation for $(typeof(d))")
error_space(d) = nothing

Evaluation(d::Space,x::EvalPtType) = Evaluation(d,x,0)
Evaluation(::Type{T},d,n...) where {T} = (error_space(d); Evaluation(T,Space(d),n...))
Evaluation(d,n...) = (error_space(d); Evaluation(Space(d),n...))
Evaluation(x::EvalPtType,k::Integer=0) = Evaluation(UnsetSpace(),x,k)

_rangespace_eval(E::ConcreteEvaluation, ::AmbiguousSpace, ::SpecialEvalPtType) = UnsetSpace()
_rangespace_eval(E::ConcreteEvaluation, ::AmbiguousSpace, ::Any) = ConstantSpace()
_rangespace_eval(E::ConcreteEvaluation, ::Space, ::Any) = ConstantSpace(Point(E.x))
function _rangespace_eval(E::ConcreteEvaluation, ::Space, ::SpecialEvalPtType)
    d = domain(domainspace(E))
    isambiguous(d) && return ConstantSpace()
    dop = boundaryfn(E.x)
    return ConstantSpace(Point(dop(d)))
end
rangespace(E::ConcreteEvaluation) = _rangespace_eval(E, E.space, evaluation_point(E))


function convert(::Type{Operator{T}},E::ConcreteEvaluation) where T
    if T == eltype(E)
        E
    else
        ConcreteEvaluation{typeof(E.space),typeof(E.x),typeof(E.order),T}(E.space,E.x,E.order)
    end
end



## default getindex
_eval(f, x) = f(x)
_eval(f, x::SpecialEvalPtType) = boundaryevalfn(x)(f)
function getindex(D::ConcreteEvaluation,k::Integer)
    T = prectype(domainspace(D))
    f = D.space(k-1)
    df = differentiate(f,D.order)
    v = _eval(df, D.x)
    strictconvert(eltype(D), v)
end

getindex(D::ConcreteEvaluation, r::AbstractRange) = [D[j] for j in r]

boundaryfn(x::typeof(rightendpoint)) = x
boundaryfn(x::typeof(leftendpoint)) = x
boundaryfn(x::Boundary) = isleftendpoint(x) ? leftendpoint : rightendpoint
boundaryevalfn(::typeof(rightendpoint)) = last
boundaryevalfn(::typeof(leftendpoint)) = first
boundaryevalfn(x::Boundary) = isleftendpoint(x) ? first : last





## EvaluationWrapper

struct EvaluationWrapper{S<:Space,M,FS<:Operator,OT,T<:Number} <: Evaluation{T}
    space::S
    x::M
    order::OT
    op::FS
end


@wrapper EvaluationWrapper false
dominantstyle(S::StyleConflict, ::Any, ::EvaluationWrapper) = S.wrapper

EvaluationWrapper(sp::Space,x,order,func::Operator) =
    EvaluationWrapper{typeof(sp),typeof(x),typeof(func),typeof(order),eltype(func)}(sp,x,order,func)


domainspace(E::Evaluation) = E.space
promotedomainspace(E::Evaluation,sp::Space) = Evaluation(sp,E.x,E.order)



function convert(::Type{Operator{T}},E::EvaluationWrapper) where T
    if T == eltype(E)
        E
    else
        EvaluationWrapper(E.space,E.x,E.order,strictconvert(Operator{T},E.op))::Operator{T}
    end
end

## Convenience routines


evaluate(d::Domain,x) = Evaluation(d,x)

"""
    ldiffbc(d::Domain, k)

The boundary condition of the `k`-th order derivative on the left endpoint of `d`. See also [`rdiffbc`](@ref), [`ldirichlet`](@ref) and [`lneumann`](@ref).
"""
ldiffbc(d,k) = Evaluation(d,leftendpoint,k)

"""
    rdiffbc(d::Domain, k)

The boundary condition of the `k`-th order derivative on the right endpoint of `d`. See also [`ldiffbc`](@ref), [`rdirichlet`](@ref) and [`rneumann`](@ref).
"""
rdiffbc(d,k) = Evaluation(d,rightendpoint,k)

"""
    ldirichlet(d::Domain) = ldiffbc(d, 0)

The dirichlet boundary condition on the left endpoint of `d`. See also [`rdirichlet`](@ref) and [`ldiffbc`](@ref).
"""
ldirichlet(d) = ldiffbc(d,0)

"""
    rdirichlet(d::Domain) = rdiffbc(d, 0)

The dirichlet boundary condition on the right endpoint of `d`. See also [`ldirichlet`](@ref) and [`rdiffbc`](@ref).
"""
rdirichlet(d) = rdiffbc(d,0)

"""
    lneumann(d::Domain) = ldiffbc(d, 1)

The neumann boundary condition on the left endpoint of `d`. See also [`rneumann`](@ref) and [`ldiffbc`](@ref).  
"""
lneumann(d) = ldiffbc(d,1)

"""
    rneumann(d::Domain) = rdiffbc(d, 1)

The neumann boundary condition on the right endpoint of `d`. See also [`lneumann`](@ref) and [`rdiffbc`](@ref).  
"""
rneumann(d) = rdiffbc(d,1)

"""
    ivp(d::Domain, k) = [ldiffbc(d,i) for i=0:k-1]
    ivp(d) = ivp(d,2)

The conditions for the `k`-th order initial value problem. See also [`ldiffbc`](@ref), [`bvp`](@ref) and [`periodic`](@ref).
"""
ivp(d,k) = [ldiffbc(d,i) for i=0:k-1]

"""
    bvp(d::Domain, k) = vcat([ldiffbc(d,i) for i=0:div(k,2)-1],
                        [rdiffbc(d,i) for i=0:div(k,2)-1])
    bvp(d) = bvp(d,2)

The conditions for the `k`-th order boundary value problem. See also [`ldiffbc`](@ref), [`rdiffbc`](@ref), [`ivp`](@ref) and [`periodic`](@ref).
"""
bvp(d,k) = vcat([ldiffbc(d,i) for i=0:div(k,2)-1],
                [rdiffbc(d,i) for i=0:div(k,2)-1])

"""
    periodic(d::Domain,k) = [ldiffbc(d,i) - rdiffbc(d,i) for i=0:k]

The conditions for the `k`-th order periodic problem. See also [`ldiffbc`](@ref), [`rdiffbc`](@ref), [`ivp`](@ref) and [`bvp`](@ref)
"""
periodic(d,k) = [ldiffbc(d,i) - rdiffbc(d,i) for i=0:k]

# shorthand for second order
ivp(d) = ivp(d,2)
bvp(d) = bvp(d,2)



for op in (:rdirichlet,:ldirichlet,:lneumann,:rneumann,:ivp,:bvp)
    @eval begin
        $op() = $op(UnsetSpace())
    end
end

for op in (:ldiffbc,:rdiffbc,:ivp,:bvp,:periodic)
    @eval $op(k::Integer) = $op(UnsetSpace(),k)
end



abstract type Dirichlet{S,T} <: Operator{T} end


struct ConcreteDirichlet{S,V,T} <: Dirichlet{S,T}
    domainspace::S
    rangespace::V
    order::Int
end

ConcreteDirichlet(sp::Space,rs::Space,order) =
    ConcreteDirichlet{typeof(sp),typeof(rs),rangetype(sp)}(sp,rs,order)
ConcreteDirichlet(sp::Space,order) = ConcreteDirichlet(sp,Space(∂(domain(sp))),order)
ConcreteDirichlet(sp::Space) = ConcreteDirichlet(sp,0)

convert(::Type{Operator{T}},B::ConcreteDirichlet{S,V}) where {S,V,T} =
    ConcreteDirichlet{S,V,T}(B.domainspace,B.rangespace,B.order)


struct DirichletWrapper{S,T} <: Dirichlet{S,T}
    op::S
    order::Int
end

@wrapper DirichletWrapper

DirichletWrapper(B::Operator,λ=0) = DirichletWrapper{typeof(B),eltype(B)}(B,λ)

convert(::Type{Operator{T}},B::DirichletWrapper) where {T} =
    DirichletWrapper(Operator{T}(B.op),B.order)::Operator{T}

# Default is to use diffbca
default_Dirichlet(sp::Space,λ) =
    DirichletWrapper(InterlaceOperator((ldiffbc(sp,λ), rdiffbc(sp,λ)), false), λ)
Dirichlet(sp::Space,λ) = default_Dirichlet(sp,λ)
Dirichlet(sp::Space) = Dirichlet(sp,0)
Dirichlet() = Dirichlet(UnsetSpace())

Dirichlet(d::Domain,λ...) = Dirichlet(Space(d),λ...)
Neumann(sp::Space) = Dirichlet(sp,1)
Neumann(d::Domain) = Neumann(Space(d))
Neumann() = Dirichlet(UnsetSpace(),1)

domainspace(B::ConcreteDirichlet) = B.domainspace
rangespace(B::ConcreteDirichlet) = B.rangespace

promotedomainspace(E::Dirichlet,sp::Space) = Dirichlet(sp,E.order)



"""
`Evaluation(sp,x,k)` is the functional associated with evaluating the
`k`-th derivative at a point `x` for the space `sp`.
"""
Evaluation(sp::Space,x,k)

"""
`Evaluation(sp,x)` is the functional associated with evaluating
at a point `x` for the space `sp`.
"""
Evaluation(sp::Space,x)


"""
`Evaluation(x)` is the functional associated with evaluating
at a point `x`.
"""
Evaluation(x)


"""
`Dirichlet(sp,k)` is the operator associated with restricting the
`k`-th derivative on the boundary for the space `sp`.
"""
Dirichlet(sp::Space,k)

"""
`Dirichlet(sp)` is the operator associated with restricting the
 the boundary for the space `sp`.
"""
Dirichlet(sp::Space)


"""
`Dirichlet()` is the operator associated with restricting on the
 the boundary.
"""
Dirichlet()




"""
`Neumann(sp)` is the operator associated with restricting the
normal derivative on the boundary for the space `sp`.
At the moment it is implemented as `Dirichlet(sp,1)`.
"""
Neumann(sp::Space)

"""
`Neumann()` is the operator associated with restricting the
normal derivative on the boundary.
"""
Neumann()
