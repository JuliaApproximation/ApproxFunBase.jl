export eigs


eigvals(A::Operator,n::Integer;tolerance::Float64=100eps()) =
    eigs(A,n;tolerance=tolerance)[1]

"""
    λ, V = eigs(A::Operator, n::Integer; tolerance::Float64=100eps())

Compute the eigenvalues and eigenvectors of the operator `A`. This is done in the following way:

* Truncate `A` into an `n×n` matrix `A₁`.
* Compute eigenvalues and eigenvectors of `A₁`.
* Filter for those eigenvectors of `A₁` that are approximately eigenvectors
of `A` as well. The `tolerance` argument controls, which eigenvectors of the approximation are kept.
"""
function eigs(A::Operator,n::Integer;tolerance::Float64=100eps())
    typ = eltype(A)

    ds=domainspace(A)
    C = Conversion(ds,rangespace(A))

    A1,C1 = zeros(typ,n,n),zeros(typ,n,n)
    A1[1:end,1:n] = A[1:n,1:n]
    C1[1:end,1:n] = C[1:n,1:n]

    λ,V=eigen(A1,C1)

    pruneeigs(λ,V,ds,tolerance)
end

eigvals(Bcs::Operator,A::Operator,n::Integer;tolerance::Float64=100eps()) =
    eigs(Bcs,A,n;tolerance=tolerance)[1]

"""
    λ, V = eigs(BC::Operator, A::Operator, n::Integer; tolerance::Float64=100eps())

Compute `n` eigenvalues and eigenvectors of the operator `A`,
subject to the boundary conditions `BC`.

# Examples
```jldoctest
julia> #= We compute the spectrum of the second derivative,
          subject to zero boundary conditions.
          We solve this eigenvalue problem in the Chebyshev basis =#

julia> S = Chebyshev();

julia> D = Derivative(S, 2);

julia> BC = Dirichlet(S);

julia> λ, v = ApproxFun.eigs(BC, D, 100);

julia> λ[1:10] ≈ [-(n*pi/2)^2 for n in 1:10] # compare with the analytical result
true
```
"""
function eigs(Bcs_in::Operator,A_in::Operator,n::Integer;tolerance::Float64=100eps())
    Bcs, A = promotedomainspace([Bcs_in, A_in])

    nf = size(Bcs,1)
    @assert isfinite(nf)

    typ = promote_type(eltype(Bcs),eltype(A))

    ds=domainspace(A)
    C = Conversion(ds,rangespace(A))

    A1,C1 = zeros(typ,n,n),zeros(typ,n,n)
    A1[1:nf,1:n] = Bcs[1:nf,1:n]
    A1[nf+1:end,1:n] = A[1:n-nf,1:n]
    C1[nf+1:end,1:n] = C[1:n-nf,1:n]

    λ,V = eigen(A1,C1)

    λ, V = pruneeigs(λ,V,ds,tolerance)
    p = sortperm(λ; lt=(x,y) -> isless(abs(x),abs(y)))
    λ[p], V[p]
end

function pruneeigs(λ,V,ds,tolerance)
    retλ=eltype(λ)[]
    retV=VFun{typeof(ds),eltype(V)}[]
    n=length(λ)
    for k=1:n
        if slnorm(V,n-3:n,k)≤tolerance
            push!(retλ,λ[k])
            f = Fun(ds, V[:,k])
            f /= norm(f)
            push!(retV, f)
        end
    end
    retλ,retV
end

abstract type EigenSystem end

"""
    SymmetricEigensystem(L::Operator, B::Operator, QuotientSpaceType::Type = QuotientSpace)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is self-adjoint with respect to the standard `L2`
inner product given the boundary conditions `B`.

!!! note
    No tests are performed to assert that the operator `L` is self-adjoint, and it's the user's responsibility
    to ensure that the operators are compliant.

The optional argument `QuotientSpaceType` specifies the type of space to be used to denote the quotient space in the basis
recombination process. In most cases, the default choice of `QuotientSpace` is a good one. In specific instances where `B`
is rank-deficient (e.g. it contains a column of zeros,
which typically happens if one of the basis elements already satiafies the boundary conditions),
one may need to choose this to be a `PathologicalQuotientSpace`.

!!! note
    No checks on the rank of `B` are carried out, and it's up to the user to specify the correct type.
"""
SymmetricEigensystem

"""
    SkewSymmetricEigensystem(L::Operator, B::Operator, QuotientSpaceType::Type = QuotientSpace)

Represent the eigensystem `L v = λ v` subject to `B v = 0`, where `L` is skew-symmetric with respect to the standard `L2`
inner product given the boundary conditions `B`.

!!! note
    No tests are performed to assert that the operator `L` is skew-symmetric, and it's the user's responsibility
    to ensure that the operators are compliant.

The optional argument `QuotientSpaceType` specifies the type of space to be used to denote the quotient space in the basis
recombination process. In most cases, the default choice of `QuotientSpace` is a good one. In specific instances where `B`
is rank-deficient (e.g. it contains a column of zeros,
which typically happens if one of the basis elements already satiafies the boundary conditions),
one may need to choose this to be a `PathologicalQuotientSpace`.

!!! note
    No checks on the rank of `B` are carried out, and it's up to the user to specify the correct type.
"""
SkewSymmetricEigensystem

for SET in (:SymmetricEigensystem, :SkewSymmetricEigensystem)
    @eval begin
        struct $SET{LT,QST} <: EigenSystem
            L :: LT
            QS :: QST

            function $SET(L, B, ::Type{QST} = QuotientSpace) where {QST}
                L2, B2 = promotedomainspace((L, B))
                if isambiguous(domainspace(L))
                    throw(ArgumentError("could not detect spaces, please specify the domain spaces for the operators"))
                end

                QS = QST(B2)
                new{typeof(L2),typeof(QS)}(L2, QS)
            end
        end
    end
end

function basis_recombination(SE::EigenSystem)
    L, QS = SE.L, SE.QS
    S = domainspace(L)
    C = Conversion(S, rangespace(L))
    C_S_NS = Conversion_normalizedspace(S)
    C_NS_S = Conversion_normalizedspace(S, Val(:backward))
    Q = Conversion(QS, S)
    R = C_S_NS * Q
    P = cache(PartialInverseOperator(C, (0, bandwidth(L, 1) + bandwidth(R, 1) + bandwidth(C, 2))))
    # A = R' * C_S_NS * P * L * C_NS_S * R
    # We use C_NS_S * R == Q to simplify this
    A = R' * C_S_NS * P * L * Q
    B = R'R

    return A, B
end

"""
    bandmatrices_eigen(S::Union{SymmetricEigensystem, SkewSymmetricEigensystem}, n::Integer)

Recast the symmetric/skew-symmetric eigenvalue problem `L v = λ v` subject to `B v = 0` to the generalized
eigenvalue problem `SA v = λ SB v`, where `SA` and `SB` are banded operators, and
return the `n × n` matrix representations of `SA` and `SB`.
If `S isa SymmetricEigensystem`, the returned matrices will be `Symmetric`.

!!! note
    No tests are performed to assert that the system is symmetric/skew-symmetric, and it's the user's responsibility
    to ensure that the operators are compliant.
"""
function bandmatrices_eigen(S::SymmetricEigensystem, n::Integer)
    A, B = _bandmatrices_eigen(S, n)
    SA = Symmetric(A, :L)
    SB = Symmetric(B, :L)
    return SA, SB
end

function bandmatrices_eigen(S::EigenSystem, n::Integer)
    A, B = _bandmatrices_eigen(S, n)
    A2 = tril(A, bandwidth(A,1))
    B2 = tril(B, bandwidth(B,1))
    Matrix(A2), Matrix(B2)
end

function _bandmatrices_eigen(S::EigenSystem, n::Integer)
    AA, BB = basis_recombination(S)
    A = AA[1:n,1:n]
    B = BB[1:n,1:n]
    return A, B
end

function eigvals(S::EigenSystem, n::Integer)
    SA, SB = bandmatrices_eigen(S, n)
    eigvals(SA, SB)
end

function eigs(Seig::EigenSystem, n::Integer; tolerance::Float64=100eps())
    SA, SB = bandmatrices_eigen(Seig, n)
    λ, v = eigen(SA, SB)
    vm = Matrix(v)
    L, QS = Seig.L, Seig.QS
    S = domainspace(L)
    Q = Conversion(QS, S)
    QM = Q[FiniteRange, axes(vm, 1)]
    V = QM * vm
    pruneeigs(λ,V,S,tolerance)
end
