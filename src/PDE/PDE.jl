export timedirichlet

include("KroneckerOperator.jl")

## PDE

lap(d::Space) = Laplacian(d)
lap(d::Domain) = Laplacian(d)
lap(f::Fun) = Laplacian()*f


function Laplacian(d::BivariateSpace,k::Integer)
    Dx2=Derivative(d, Vec{2}(2,0))
    Dy2=Derivative(d, Vec{2}(0,2))
    if k==1
        LaplacianWrapper(Dx2+Dy2,d,k)
    else
        @assert k > 0
        Δ=Laplacian(d,1)
        LaplacianWrapper(TimesOperator(Laplacian(rangespace(Δ),k-1),Δ),k)
    end
end

Laplacian(d::EuclideanDomain{2}, k::Integer) = Laplacian(Space(d),k)
grad(d::ProductDomain) = grad(Space(d))
function grad(d::BivariateSpace)
    n = length(factors(d))
    @assert n == 2 "grad for n>2 is not implemented"
    Vec{2}(Derivative(d, Vec{2}(1,0)), Derivative(d, Vec{2}(0,1)))
end
grad(f::Fun{<:BivariateSpace}) = grad(space(f)) * f

function tensor_Dirichlet(d::Union{ProductDomain,TensorSpace},k)
    @assert nfactors(d)==2

    DirichletWrapper(
        if isempty(∂(factor(d,1)))
            I ⊗ Dirichlet(factor(d,2),k)
        elseif isempty(∂(factor(d,2)))
            Dirichlet(factor(d,1),k) ⊗ I
        else
            [Dirichlet(factor(d,1),k) ⊗ I;I ⊗ Dirichlet(factor(d,2),k)]
        end, k)
end

Dirichlet(d::Union{ProductDomain,TensorSpace},k) = tensor_Dirichlet(d,k)


function timedirichlet(d::Union{ProductDomain,TensorSpace})
    @assert nfactors(d)==2
    Bx=Dirichlet(factor(d,1))
    Bt=ldirichlet(factor(d,2))
    [I⊗Bt;Bx⊗I]
end


# Operators on a univariate space act on the coefficient Funs when multiplied from the left,
# and on the basis of the second space when multiplied from the right.
# Note that the latter produces a function, and not an operator. To obtain an operator,
# right multiply by a kronecker product of operators
# We re-route through _mulop to distinguish between operators on UnivariateSpace and
# those on BivariateSpace
function _mulop(B::Operator, ::UnivariateSpace, f::ProductFun)
    if isafunctional(B)
        Fun(factor(space(f),2),map(c->Number(B*c),f.coefficients))
    else
        ProductFun(map(c->B*c,f.coefficients), space(f))
    end
end
*(B::Operator,f::ProductFun) = _mulop(B, domainspace(B), f)

_mulop(f::ProductFun, ::UnivariateSpace, B::Operator) = transpose(B*(transpose(f)))
*(f::ProductFun,B::Operator) = _mulop(f, domainspace(B), B)
