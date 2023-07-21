## Root finding for Chebyshev expansions
#
#  Contains code that is based in part on Chebfun v5's chebfun/@chebteck/roots.m,
# which is distributed with the following license:

# Copyright (c) 2015, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


function complexroots(f::Fun{C}) where C<:Chebyshev
    if ncoefficients(f)==0 || (ncoefficients(f)==1 && isapprox(f.coefficients[1],0))
        throw(ArgumentError("Tried to take roots of a zero function."))
    elseif ncoefficients(f)==1
        ComplexF64[]
    elseif ncoefficients(f)==2
        ComplexF64[-f.coefficients[1]/f.coefficients[2]]
    else
        fromcanonical.(f,colleague_eigvals(f.coefficients))
    end
end

function roots(f::Fun{<:Chebyshev})
    g = Fun(space(f), float.(coefficients(f)))
    gg = setcanonicaldomain(g)
    r = roots(gg)
    fromcanonical.(f,r)
end


for (BF,FF) in ((BigFloat,Float64),(Complex{BigFloat},ComplexF64))
    @eval function roots( f::Fun{C,$BF} ) where C<:Chebyshev
    # FIND THE ROOTS OF AN IFUN.

        d = domain(f)
        c = f.coefficients
        vscale = maximum(abs,values(f))
        if vscale == 0
            throw(ArgumentError("Tried to take roots of a zero function."))
        end

        hscale = maximum( [abs(leftendpoint(d)), abs(rightendpoint(d))] )
        htol = eps(2000.)*max(hscale, 1)  # TODO: choose tolerance better

        # calculate Flaot64 roots
        r = Array{$BF}(rootsunit_coeffs(strictconvert(Vector{$FF},c./vscale), Float64(htol)))
        # Map roots from [-1,1] to domain of f:
        rts = fromcanonical.(Ref(d),r)
        fp = differentiate(f)

        # do Newton 3 time
        for _ = 1:3
            rts .-=extrapolate.(Ref(f),rts)./extrapolate.(Ref(fp),rts)
        end

        return rts
    end
end


function roots( f::Fun{C,TT} ) where {C<:Chebyshev,TT<:Union{Float64,ComplexF64}}
# FIND THE ROOTS OF AN IFUN.
    if iszero(f)
        throw(ArgumentError("Tried to take roots of a zero function."))
    end

    d = domain(f)
    c = f.coefficients
    vscale = maximum(abs,values(f))

    hscale = max(abs(leftendpoint(d)), abs(rightendpoint(d)) )
    htol = eps(2000.)*max(hscale, 1)  # TODO: choose tolerance better


    cvscale=c./vscale
    r = rootsunit_coeffs(cvscale, Float64(htol))

    # Check endpoints, as these are prone to inaccuracy
    # which can be deadly.
    if (isempty(r) || !isapprox(last(r),1.)) && abs(sum(cvscale)) < htol
        push!(r,1.)
    end
    if (isempty(r) || !isapprox(first(r),-1.)) && abs(alternatingsum(cvscale)) < htol
        insert!(r,1,-1.)
    end


    # Map roots from [-1,1] to domain of f:
    return fromcanonical.(Ref(d),r)
end



function colleague_matrix( c::Vector{T} ) where T<:Number
#TODO: This is command isn't typed correctly
# COMPUTE THE ROOTS OF A LOW DEGREE POLYNOMIAL BY USING THE COLLEAGUE MATRIX:
    c = chop(c,0) # prune exact zeros
    n = length(c) - 1
    A=zeros(T,n,n)

    for k=1:n-1
        A[k+1,k]=0.5
        A[k,k+1]=0.5
    end
    for k=1:n
        A[1,end-k+1]-=0.5*c[k]/c[end]
    end
    A[n,n-1]=one(T)

    # TODO: can we speed things up because A is lower-Hessenberg
    # Standard colleague matrix (See [Good, 1961]):
    A
end


function colleague_balance!(M)
    n=size(M,1)
    if n<2
        return M
    end

    d=1+abs(M[1,2])
    M[1,2]/=d;M[2,1]*=d

    for k=3:n
        M[k-1,k]*=d;M[k,k-1]/=d
        d+=abs(M[1,k])
        M[1,k]/=d;M[k-1,k]/=d;M[k,k-1]*=d
    end
    M
end


# colleague_eigvals( c::Vector )=hesseneigvals(colleague_balance!(colleague_matrix(c)))

colleague_eigvals( c::Vector )=eigvals(colleague_matrix(c))

function PruneOptions( r, htol::Float64 )
# ONLY KEEP ROOTS IN THE INTERVAL

    # Remove dangling imaginary parts:
    r = real( r[ abs.(imag.(r)) .< htol ] )
    # Keep roots inside [-1 1]:
    r = sort( r[ abs.(r) .<= 1+htol ] )
    # Put roots near ends onto the domain:
    r = min.( max.( r, -1 ), 1)

    # Return the pruned roots:
    return r
end

rootsunit_coeffs(c::Vector{T}, htol::Float64) where {T<:Number} =
    rootsunit_coeffs(c,htol,ClenshawPlan(T,Chebyshev(),length(c),length(c)))
function rootsunit_coeffs(c::Vector{T}, htol::Float64,clplan::ClenshawPlan{S,T}) where {S,T<:Number}
# Computes the roots of the polynomial given by the coefficients c on the unit interval.


    # If recursive subdivision is used, then subdivide [-1,1] into [-1,splitPoint] and [splitPoint,1].
    splitPoint = -0.004849834917525

    # Simplify the coefficients by chopping off the tail:
    nrmc=norm(c, 1)
    @assert nrmc > 0  # There's an error if we make it this far
    c = chop(c,eps()*nrmc)
    n = length(c)

    if n == 0

        # EMPTY FUNCTION
        r = Float64[]

    elseif n == 1

        # CONSTANT FUNCTION
        r = ( c[1] == 0.0 ) ? Float64[0.0] : Float64[]

    elseif n == 2

        # LINEAR POLYNOMIAL
        r1 = -c[1]/c[2];
        r = ( (abs(imag(r1))>htol) | (abs(real(r1))>(1+htol)) ) ? Float64[] : Float64[max(min(real(r1),1),-1)]

    elseif n <= 70

        # COLLEAGUE MATRIX
        # The recursion subdividing below will keep going until we have a piecewise polynomial
        # of degree at most 50 and we likely end up here for each piece.

        # Adjust the coefficients for the colleague matrix
        # Prune roots depending on preferences:
        r = PruneOptions( colleague_eigvals(c), htol )::Vector{Float64}
    else

        #  RECURSIVE SUBDIVISION:
        # If n > 50, then split the interval [-1,1] into [-1,splitPoint], [splitPoint,1].
        # Find the roots of the polynomial on each piece and then concatenate. Recurse if necessary.

        # Evaluate the polynomial at Chebyshev grids on both intervals:
        #(clenshaw! overwrites points, which only makes sense if c is real)

        v1 = if isa(c, Vector{Float64})
            clenshaw!(c, convert(Vector, points(-1..splitPoint,n)), clplan)
        else
            clenshaw(c, points(-1..splitPoint,n),clplan)
        end
        v2 = if isa(c, Vector{Float64})
            clenshaw!(c, convert(Vector, points(splitPoint..1 ,n)), clplan)
        else
            clenshaw(c, points(splitPoint..1 ,n),clplan)
        end

        # Recurse (and map roots back to original interval):
        p = plan_chebyshevtransform( v1 )
        r = [ (splitPoint - 1)/2 .+ (splitPoint + 1)/2 .* rootsunit_coeffs( p*v1, 2*htol,clplan) ;
                 (splitPoint + 1)/2 .+ (1 - splitPoint)/2 .* rootsunit_coeffs( p*v2, 2*htol,clplan) ]

    end

    # Return the roots:
    r
end


extremal_args(f::Fun{S}) where {S<:ContinuousSpace} = cat(1,[extremal_args(fp) for fp in components(f)]..., dims=1)

for op in (:(maximum),:(minimum))
    @eval begin

        $op(f::Fun{ContinuousSpace{T,R},T}) where {T<:Real,R<:Real} =
            $op(map($op,components(f)))
        $op(::typeof(abs), f::Fun{ContinuousSpace{T,R},T}) where {T<:Real,R<:Real} =
            $op(abs, map(g -> $op(abs, g),components(f)))
    end
end

extrema(f::Fun{ContinuousSpace{T,R},T}) where {T<:Real,R<:Real} =
    mapreduce(extrema,(x,y)->extrema([x...;y...]),components(f))


function roots(f::Fun{P}) where P<:ContinuousSpace
    rts=mapreduce(roots,vcat,components(f))
    k=1
    while k < length(rts)
        if isapprox(rts[k],rts[k+1])
            rts=rts[[1:k;k+2:end]]
        else
            k+=1
        end
    end

    rts
end



