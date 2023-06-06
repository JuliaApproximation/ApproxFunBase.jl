export Derivative,Integral,Laplacian,Volterra


abstract type CalculusOperator{S,OT,T}<:Operator{T} end


## Note that all functions called in calculus_operator must be exported

macro calculus_operator(Op)
    ConcOp = Symbol(:Concrete, Op)
    WrappOp = Symbol(Op, :Wrapper)
    DefaultOp = Symbol(:Default, Op)
    q = quote
        # The SSS, TTT are to work around #9312
        abstract type $Op{SSS,OT,TTT} <: ApproxFunBase.CalculusOperator{SSS,OT,TTT} end

        struct $ConcOp{S<:Space,OT,T} <: $Op{S,OT,T}
            space::S        # the domain space
            order::OT
        end
        struct $WrappOp{BT<:Operator,S<:Space,R<:Space,OT,T} <: $Op{S,OT,T}
            op::BT
            order::OT
            domainspace::S
            rangespace::R
        end

        ApproxFunBase.@wrapper $WrappOp false false
        ApproxFunBase.domainspace(A::$WrappOp) = A.domainspace
        ApproxFunBase.rangespace(A::$WrappOp) = A.rangespace

        ## Constructors
        $ConcOp(sp::ApproxFunBase.Space,k) = $ConcOp{typeof(sp),typeof(k),ApproxFunBase.prectype(sp)}(sp,k)

        $Op(sp::ApproxFunBase.UnsetSpace,k) = $ConcOp(sp,k)
        $Op(sp::ApproxFunBase.UnsetSpace,k::Number) = $ConcOp(sp,k)
        $Op(sp::ApproxFunBase.UnsetSpace,k::Real) = $ConcOp(sp,k)
        $Op(sp::ApproxFunBase.UnsetSpace,k::Integer) = $ConcOp(sp,k)

        function $DefaultOp(sp::ApproxFunBase.Space, k)
            csp=ApproxFunBase.canonicalspace(sp)
            if ApproxFunBase.conversion_type(csp,sp)==csp   # Conversion(sp,csp) is not banded, or sp==csp
               error("Implement $(string($Op))($(string(sp)),$k)")
            end
            O = $Op(csp,k)
            C = ApproxFunBase.Conversion_maybeconcrete(sp, csp, Val(:forward))
            Top = ApproxFunBase.TimesOperator([O,C])
            $WrappOp(Top, sp, k, ApproxFunBase.rangespace(O))
        end

        $DefaultOp(d,k) = $Op(ApproxFunBase.Space(d),k)

        $DefaultOp(sp) = $Op(sp,1)
        $DefaultOp() = $Op(ApproxFunBase.UnsetSpace())
        $DefaultOp(k::Number) = $Op(ApproxFunBase.UnsetSpace(),k)
        $DefaultOp(k::AbstractVector) = $Op(ApproxFunBase.UnsetSpace(),k)

        $Op(x...) = $DefaultOp(x...)
        $ConcOp(S::ApproxFunBase.Space) = $ConcOp(S,1)

        function Base.convert(::Type{ApproxFunBase.Operator{T}},D::$ConcOp) where T
            if T==eltype(D)
                D
            else
                $ConcOp{typeof(D.space),typeof(D.order),T}(D.space, D.order)
            end
        end

        $WrappOp(op::ApproxFunBase.Operator, order = 1, d = domainspace(op), r = rangespace(op)) =
            $WrappOp{typeof(op),typeof(d),typeof(r),typeof(order),eltype(op)}(op,order,d,r)

        function Base.convert(::Type{ApproxFunBase.Operator{T}},D::$WrappOp) where T
            if T==eltype(D)
                D
            else
                op=ApproxFunBase.strictconvert(ApproxFunBase.Operator{T},D.op)
                S = ApproxFunBase.domainspace(D)
                R = ApproxFunBase.rangespace(D)
                $WrappOp(op,D.order,S,R)::ApproxFunBase.Operator{T}
            end
        end

        ## Routines
        ApproxFunBase.domainspace(D::$ConcOp) = D.space

        Base.getindex(::$ConcOp{ApproxFunBase.UnsetSpace,OT,T},k::Integer,j::Integer) where {OT,T} =
            error("Spaces cannot be inferred for operator")

        ApproxFunBase.rangespace(D::$ConcOp{ApproxFunBase.UnsetSpace,T}) where {T} = UnsetSpace()

        #promoting domain space is allowed to change range space
        # for integration, we fall back on existing conversion for now
        ApproxFunBase.promotedomainspace(D::$Op, sp::ApproxFunBase.UnsetSpace) = D


        function ApproxFunBase.promotedomainspace(D::$Op, sp::ApproxFunBase.Space)
            if ApproxFunBase.isambiguous(domain(sp))
                $Op(typeof(sp)(domain(D)),D.order)
            else
                $Op(sp,D.order)
            end
        end
    end

    return esc(q)
end

choosedomainspace(M::CalculusOperator{UnsetSpace},sp::Space) =
    iswrapper(M) ? choosedomainspace(M.op,sp) : sp  # we assume the space itself will work



@calculus_operator(Derivative)
@calculus_operator(Integral)
@calculus_operator(Volterra)








## Overrideable


## Convenience routines


integrate(d::Domain) = Integral(d,1)


# Default is to use ops
differentiate(f::Fun)=Derivative(space(f))*f
function integrate(f::Fun)
    d=domain(f)
    cd=canonicaldomain(d)
    if typeof(d)==typeof(cd)  || isperiodic(d)
        Integral(space(f))*f
    else
        # map to canonical domain
        setdomain(integrate(setdomain(f,cd)*fromcanonicalD(f)),d)
    end
end

function Base.sum(f::Fun)
    d=domain(f)
    cd=canonicaldomain(d)
    if typeof(cd)==typeof(d)  || isperiodic(d)
        g=integrate(f)
        last(g)-first(g)
    else
        # map first
        sum(setdomain(f,cd)*fromcanonicalD(f))
    end
end

function linesum(f::Fun)
    cd=canonicaldomain(f)
    d=domain(f)

    if isreal(d)
        a,b=leftendpoint(d),rightendpoint(d)
        sign(last(b)-first(a))*sum(f)
    elseif typeof(cd)==typeof(d)  || isperiodic(d)
        error("override linesum for $(f.space)")
    else
        # map first
        linesum(setdomain(f,cd)*abs(fromcanonicalD(f)))
    end
end

∫(f::Fun)=integrate(f)

⨜(f::Fun)=cumsum(f)

for OP in (:Σ,:∮,:⨍,:⨎)
    @eval $OP(f::Fun)=sum(f)
end


# Multivariate



@calculus_operator(Laplacian)


## Map to canonical
@inline function _DefaultDerivative(sp::Space, k::Number)
    assert_integer(k)
    if nameof(typeof(canonicaldomain(sp))) == nameof(typeof(domain(sp)))
        # this is the normal default constructor
        csp=canonicalspace(sp)
        if conversion_type(csp,sp)==csp   # Conversion(sp,csp) is not banded, or sp==csp
           error("Implement Derivative(", sp, ",", k,")")
        end
        D = Derivative(csp,k)
        C = Conversion_maybeconcrete(sp, csp, Val(:forward))
        DerivativeWrapper(TimesOperator(D, C), k, sp, rangespace(D))
    else
        csp = canonicalspace(sp)
        D1 = if csp == sp
            _Dsp = invfromcanonicalD(sp)*Derivative(setdomain(sp,canonicaldomain(sp)))
            rsp = rangespace(_Dsp)
            _Dsp
        else
            Dcsp = Derivative(csp)
            rsp = rangespace(Dcsp)
            Dcsp * Conversion_maybeconcrete(sp, csp, Val(:forward))
        end
        D=DerivativeWrapper(SpaceOperator(D1,sp,setdomain(rsp,domain(sp))),1)
        if k==1
            D
        else
            DerivativeWrapper(TimesOperator(Derivative(rangespace(D),k-1),D), k, sp)
        end
    end
end

@static if VERSION >= v"1.8"
    for f in (:DefaultDerivative, :DefaultIntegral)
        _f = Symbol(:_, f)
        @eval Base.@constprop :aggressive $f(sp::Space, k::Number) = $_f(sp, k)
    end
else
    for f in (:DefaultDerivative, :DefaultIntegral)
        _f = Symbol(:_, f)
        @eval $f(sp::Space, k::Number) = $_f(sp, k)
    end
end

@inline function _DefaultIntegral(sp::Space, k::Number)
    assert_integer(k)
    if nameof(typeof(canonicaldomain(sp))) == nameof(typeof(domain(sp)))
        # this is the normal default constructor
        csp=canonicalspace(sp)
        if conversion_type(csp,sp)==csp   # Conversion(sp,csp) is not banded, or sp==csp
            # we require that Integral is overridden
            error("Implement Integral($(string(sp)),$k)")
        end
        Ik = Integral(csp,k)
        C = Conversion_maybeconcrete(sp, csp, Val(:forward))
        IntegralWrapper(TimesOperator(Ik, C), k, sp, rangespace(Ik))
    elseif k > 1
        Q=Integral(sp,1)
        IntegralWrapper(TimesOperator(Integral(rangespace(Q),k-1),Q),k)
    else # k==1
        csp=setdomain(sp,canonicaldomain(sp))

        x=Fun(identity,domain(csp))
        M=Multiplication(fromcanonicalD(sp,x),csp)
        Q=Integral(rangespace(M))*M
        IntegralWrapper(SpaceOperator(Q,sp,setdomain(rangespace(Q),domain(sp))),1)
    end
end


for TYP in (:Derivative,:Integral,:Laplacian)
    @eval begin
        function *(D1::$TYP,D2::$TYP)
            @assert domain(D1) == domain(D2)

            $TYP(domainspace(D2),D1.order+D2.order)
        end
    end
end

==(A::Derivative, B::Derivative) = A.order == B.order && domainspace(A) == domainspace(B)


"""
    Derivative(sp::Space, k::Int)

Return the `k`-th derivative operator on the space `sp`.

# Examples
```jldoctest
julia> Derivative(Chebyshev(), 2) * Fun(x->x^4) ≈ Fun(x->12x^2)
true
```
"""
Derivative(::Space,::Int)

"""
    Derivative(sp::Space, k::AbstractVector{Int})

Return a partial derivative operator on a multivariate space. For example,
```julia
Dx = Derivative(Chebyshev()^2,[1,0]) # ∂/∂x
Dy = Derivative(Chebyshev()^2,[0,1]) # ∂/∂y
```

!!! tip
    Using a static vector as the second argument would help with type-stability.

# Examples
```jldoctest
julia> ∂y = Derivative(Chebyshev()^2, [0,1]);

julia> ∂y * Fun((x,y)->x^2 + y^2) ≈ Fun((x,y)->2y)
true
```
"""
Derivative(::Space, ::AbstractVector{Int})

"""
    Derivative(sp::Space)

Return the first derivative operator, equivalent to `Derivative(sp,1)`.

# Examples
```jldoctest
julia> Derivative(Chebyshev()) * Fun(x->x^2) ≈ Fun(x->2x)
true
```
"""
Derivative(::Space)

"""
    Derivative(k)

Return the `k`-th derivative, acting on an unset space.
Spaces will be inferred when applying or manipulating the operator.
If `k` is an `Int`, this returns a derivative in an univariate space.
If `k` is an `AbstractVector{Int}`, this returns a partial derivative
in a multivariate space.

# Examples
```jldoctest
julia> Derivative(1) * Fun(x->x^2) ≈ Fun(x->2x)
true

julia> Derivative([0,1]) * Fun((x,y)->x^2+y^2) ≈ Fun((x,y)->2y)
true
```
"""
Derivative(k)

"""
    Derivative()

Return the first derivative on an unset space.
Spaces will be inferred when applying or manipulating the operator.

# Examples
```jldoctest
julia> Derivative() * Fun(x->x^2) ≈ Fun(x->2x)
true
```
"""
Derivative()


"""
    Integral(sp::Space, k::Int)

Return the `k`-th integral operator on `sp`.
There is no guarantee on normalization.
"""
Integral(::Space, ::Int)


"""
    Integral(sp::Space)

Return the first integral operator, equivalent to `Integral(sp,1)`.
"""
Integral(::Space)

"""
    Integral(k::Int)

Return the `k`-th integral operator, acting on an unset space.
Spaces will be inferred when applying or manipulating the operator.
"""
Integral(k::Int)

"""
    Intergral()

Return the first integral operator on an unset space.
Spaces will be inferred when applying or manipulating the operator.
"""
Integral()

"""
    Laplacian(sp::Space)

Return the laplacian operator on space `sp`.
"""
Laplacian(::Space)

"""
    Laplacian()

Return the laplacian operator on an unset space.
Spaces will be inferred when applying or manipulating the operator.
"""
Laplacian()


const 𝒟 = Derivative()
const Δ = Laplacian()
