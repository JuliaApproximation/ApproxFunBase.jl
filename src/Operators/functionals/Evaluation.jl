export Evaluation,ivp,bvp,Dirichlet,Neumann

## Evaluation constructors

abstract type Evaluation{T}<:Operator{T} end

@functional Evaluation

# M = leftendpoint/rightendpoint if endpoint
struct ConcreteEvaluation{S<:Space,M,OT,T} <: Evaluation{T}
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

Evaluation(d::Space,x::Union{Number,typeof(leftendpoint),typeof(rightendpoint)}) = Evaluation(d,x,0)
Evaluation(::Type{T},d::Space,n...) where {T} = error("Override Evaluation for $(typeof(d))")
Evaluation(::Type{T},d,n...) where {T} = Evaluation(T,Space(d),n...)
Evaluation(S::Space,n...) = error("Override Evaluation for $(typeof(S))")
Evaluation(d,n...) = Evaluation(Space(d),n...)
Evaluation(x::Union{Number,typeof(leftendpoint),typeof(rightendpoint)}) = Evaluation(UnsetSpace(),x,0)
Evaluation(x::Union{Number,typeof(leftendpoint),typeof(rightendpoint)},k::Integer) =
    Evaluation(UnsetSpace(),x,k)

rangespace(E::ConcreteEvaluation{<:AmbiguousSpace}) = ConstantSpace()
rangespace(E::ConcreteEvaluation) = ConstantSpace(Point(E.x))

function ConcreteEvaluation{S,M,OT,T}(C::ConcreteEvaluation) where {S<:Space,M,OT,T}
    ConcreteEvaluation{S,M,OT,T}(strictconvert(S, C.space), strictconvert(M, C.x), strictconvert(OT,C.order))
end
function Operator{T}(E::ConcreteEvaluation) where T
    ConcreteEvaluation{typeof(E.space),typeof(E.x),typeof(E.order),T}(E.space,E.x,E.order)
end



## default getindex
function getindex(D::ConcreteEvaluation,k::Integer)
    T = prectype(domainspace(D))
    f = Fun(D.space, [zeros(T,k-1); one(T)])
    df = differentiate(f,D.order)
    v = df(D.x)
    strictconvert(eltype(D), v)
end

#special leftendpoint/rightendpoint overrides
for (dop, fop) in ((:leftendpoint,:first), (:rightendpoint,:last))
    @eval begin
        rangespace(E::ConcreteEvaluation{<:AmbiguousSpace,typeof($dop)}) = UnsetSpace()
        function rangespace(E::ConcreteEvaluation{<:Space,typeof($dop)})
            d = domain(domainspace(E))
            isambiguous(d) && return ConstantSpace()
            return ConstantSpace(Point($dop(d)))
        end
        function getindex(D::ConcreteEvaluation{<:Space,typeof($dop)},k::Integer)
            P = prectype(domainspace(D))
            R = eltype(D)
            R($fop(differentiate(Fun(D.space,[zeros(P,k-1);one(P)]),D.order)))
        end
    end
end






## EvaluationWrapper

struct EvaluationWrapper{S<:Space,M,FS<:Operator,OT,T<:Number} <: Evaluation{T}
    space::S
    x::M
    order::OT
    op::FS
end


@wrapper EvaluationWrapper false
EvaluationWrapper(sp::Space,x,order,func::Operator) =
    EvaluationWrapper{typeof(sp),typeof(x),typeof(func),typeof(order),eltype(func)}(sp,x,order,func)


domainspace(E::Evaluation) = E.space
promotedomainspace(E::Evaluation,sp::Space) = Evaluation(sp,E.x,E.order)


function EvaluationWrapper{S,M,FS,OT,T}(E::EvaluationWrapper) where {S<:Space,M,FS<:Operator,OT,T<:Number}
    EvaluationWrapper{S,M,FS,OT,T}(strictconvert(S, E.space), strictconvert(M, E.x),
        strictconvert(OT, E.order), strictconvert(FS, E.op))
end
function Operator{T}(E::EvaluationWrapper) where T
    EvaluationWrapper(E.space,E.x,E.order,strictconvert(Operator{T},E.op))::Operator{T}
end

## Convenience routines


evaluate(d::Domain,x) = Evaluation(d,x)
ldiffbc(d,k) = Evaluation(d,leftendpoint,k)
rdiffbc(d,k) = Evaluation(d,rightendpoint,k)

ldirichlet(d) = ldiffbc(d,0)
rdirichlet(d) = rdiffbc(d,0)
lneumann(d) = ldiffbc(d,1)
rneumann(d) = rdiffbc(d,1)


ivp(d,k) = Operator{prectype(d)}[ldiffbc(d,i) for i=0:k-1]
bvp(d,k) = vcat(Operator{prectype(d)}[ldiffbc(d,i) for i=0:div(k,2)-1],
                Operator{prectype(d)}[rdiffbc(d,i) for i=0:div(k,2)-1])

periodic(d,k) = Operator{prectype(d)}[Evaluation(d,leftendpoint,i)-Evaluation(d,rightendpoint,i) for i=0:k]

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


struct ConcreteDirichlet{S<:Space,V<:Space,T} <: Dirichlet{S,T}
    domainspace::S
    rangespace::V
    order::Int
end

ConcreteDirichlet(sp::Space,rs::Space,order) =
    ConcreteDirichlet{typeof(sp),typeof(rs),rangetype(sp)}(sp,rs,order)
ConcreteDirichlet(sp::Space,order) = ConcreteDirichlet(sp,Space(∂(domain(sp))),order)
ConcreteDirichlet(sp::Space) = ConcreteDirichlet(sp,0)

function ConcreteDirichlet{S,V,T}(C::ConcreteDirichlet) where {S<:Space,V<:Space,T}
    ConcreteDirichlet{S,V,T}(strictconvert(S, C.domainspace), strictconvert(V, C.rangespace), C.order)
end
Operator{T}(B::ConcreteDirichlet{S,V}) where {S,V,T} =
    ConcreteDirichlet{S,V,T}(B.domainspace,B.rangespace,B.order)


struct DirichletWrapper{S,T} <: Dirichlet{S,T}
    op::S
    order::Int
end

@wrapper DirichletWrapper

DirichletWrapper(B::Operator,λ=0) = DirichletWrapper{typeof(B),eltype(B)}(B,λ)

function DirichletWrapper{S,T}(D::DirichletWrapper) where {S,T}
    DirichletWrapper{S,T}(strictconvert(S, D.op), D.order)
end
Operator{T}(B::DirichletWrapper) where {T} =
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
