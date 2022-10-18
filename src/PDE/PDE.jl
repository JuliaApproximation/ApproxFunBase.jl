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
        LaplacianWrapper(Dx2+Dy2,k)
    else
        @assert k > 0
        Δ=Laplacian(d,1)
        LaplacianWrapper(TimesOperator(Laplacian(rangespace(Δ),k-1),Δ),k)
    end
end

Laplacian(d::EuclideanDomain{2}, k::Integer) = Laplacian(Space(d),k)
grad(d::ProductDomain) = [Derivative(d,[1,0]),Derivative(d,[0,1])]


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


# Operators on an univariate space may act on the second space of the ProductFun,
# consistent with treating it as an expansion in the second space
# We re-route through _mulop to distinguish between operators on UnivariateSpace and
# those on BivariateSpace
_mulop(O::Operator, ::Space, ::ProductFun) = error("define $(typeof(O)) * ProductFun")
function _mulop(B::Operator, ::UnivariateSpace, f::ProductFun)
    if isafunctional(B)
        Fun(factor(space(f),2),map(c->Number(B*c),f.coefficients))
    else
        ProductFun(space(f),map(c->B*c,f.coefficients))
    end
end
*(B::Operator,f::ProductFun) = _mulop(B, domainspace(B), f)

*(f::ProductFun,B::Operator) = transpose(B*(transpose(f)))
