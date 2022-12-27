## abs
splitmap(g,d::Domain,pts) = Fun(g,split(d , pts))

function split(d::IntervalOrSegment, pts)
    a,b = endpoints(d)
    isendpoint = all(p -> p ≈ a || p ≈ b, pts)
    isendpoint && return d

    @assert all(in(d), pts)
    PiecewiseSegment(sort!(union(endpoints(d), pts)))
end

function split(d::PiecewiseSegment, pts)
    @assert all(in(d), pts)
    PiecewiseSegment(sort!(union(d.points, pts)))
end

split(d::SegmentDomain, pts) = d


function splitatroots(f::Fun)
    d=domain(f)
    pts=union(roots(f)) # union removes multiplicities
    splitmap(f,d,pts)
end

function abs(f::Fun{<:RealUnivariateSpace,<:Real})
    d=domain(f)
    T = cfstype(f)
    pts = iszero(f) ? T[] : roots(f)
    splitmap(abs∘f,d,pts)
end

function abs(f::Fun)

    d=domain(f)

    pts = iszero(f) ? cfstype(f)[] : roots(f)

    if isempty(pts)
        # This makes sure Laurent returns real type
        real(Fun(abs∘f,space(f)))
    else
        splitmap(abs∘f,d,pts)
    end
end


midpoints(d::IntervalOrSegment) = [mean(d)]
midpoints(d::Union{UnionDomain,PiecewiseSegment}) = mapreduce(midpoints,vcat,components(d))

_UnionDomainIfMultiple(d::IntervalOrSegment) = d
_UnionDomainIfMultiple(d) = UnionDomain(components(d))

for OP in (:sign,:angle)
    @eval function $OP(f::Fun{<:RealUnivariateSpace,<:Real})
        d=domain(f)
        T = cfstype(f)
        pts = iszero(f) ? T[] : roots(f)

        if isempty(pts)
            $OP(first(f))*one(f)
        else
            d = split(d , pts)
            midpts = midpoints(d)
            Fun(_UnionDomainIfMultiple(d), $OP.(f.(midpts)))
        end
    end
end

for op in (:(max),:(min))
    @eval begin
        function $op(f::Fun{<:RealUnivariateSpace,<:Real},
                g::Fun{<:RealUnivariateSpace,<:Real})
            h=f-g
            d=domain(h)
            pts=iszero(h) ? cfstype(h)[] : roots(h)
            splitmap(x->$op(f(x),g(x)),d,pts)
        end
        $op(f::Fun{<:RealUnivariateSpace,<:Real},g::Real) = $op(f,Fun(g,domain(f)))
        $op(f::Real,g::Fun{<:RealUnivariateSpace,<:Real}) = $op(Fun(f,domain(g)),g)
    end
end



isfinite(f::Fun) = isfinite(maximum(abs,f)) && isfinite(minabs(f))

# division by fun

function /(c::Fun,f::Fun)
    d=domain(f)
    @assert domain(c) == d
    cd = canonicaldomain(f)
    if typeof(d)!=typeof(cd)
        # project first to simplify
        return setdomain(setdomain(c,cd)/setdomain(f,cd),d)
    end

    r=roots(f)
    tol=10eps(promote_type(cfstype(c),cfstype(f)))
    if length(r)==0 || norm(c.(r))<tol
        \(Multiplication(f,space(c)),c;tolerance=tol)
    else
        c*(1/f)
    end
end

function /(c::Number,f::Fun)
    r=roots(f)
    tol=10eps(promote_type(typeof(c),cfstype(f)))
    @assert length(r)==0
    \(Multiplication(f,space(f)),c*one(f);tolerance=tol)
end


# Default is just try solving ODE
function ^(f::Fun, β)
    A=Derivative()-β*differentiate(f)/f
    B=Evaluation(leftendpoint(domain(f)))
    [B;A]\[first(f)^β;0]
end

sqrt(f::Fun) = f^0.5
cbrt(f::Fun) = f^(1/3)

## We use \ as the Fun constructor might miss isolated features

## First order functions


log(f::Fun) = cumsum(differentiate(f)/f)+log(first(f))

acos(f::Fun)=cumsum(-f'/sqrt(1-f^2))+acos(first(f))
asin(f::Fun)=cumsum(f'/sqrt(1-f^2))+asin(first(f))

## Second order functions
sin(f::Fun{S,T}) where {S<:RealSpace,T<:Real} = imag(exp(im*f))
cos(f::Fun{S,T}) where {S<:RealSpace,T<:Real} = real(exp(im*f))

atan(f::Fun)=cumsum(f'/(1+f^2))+atan(first(f))

function _specialfunctionnormalizationpoint(op,growth,f)
    g=chop(growth(f),eps(cfstype(f)))
    d = domain(g)
    T = eltype(d)
    xmin = isempty(g.coefficients) ? leftendpoint(d) : T(argmin(g))::T
    xmax = isempty(g.coefficients) ? rightendpoint(d) : T(argmax(g))::T
    opfxmin,opfxmax = op(f(xmin)),op(f(xmax))
    opmax = maximum(abs,(opfxmin,opfxmax))
    xmin, xmax, opfxmin, opfxmax, opmax
end

# this is used to find a point in which to impose a boundary
# condition in calculating secial functions
function specialfunctionnormalizationpoint(op,growth,f)
    xmin, xmax, opfxmin, opfxmax, opmax = _specialfunctionnormalizationpoint(op,growth,f)
    if abs(opfxmin) == opmax
        xmax,opfxmax = xmin,opfxmin
    end
    xmax,opfxmax,opmax
end

# ODE gives the first order ODE a special function op satisfies,
# RHS is the right hand side
# growth says what to use to choose a good point to impose an initial condition
for (op, ODE, RHS, growth) in ((:(exp),    "D-f'",           "0",        :(real)),
                            (:(asinh),  "sqrt(f^2+1)*D",     "f'",       :(real)),
                            (:(acosh),  "sqrt(f^2-1)*D",     "f'",       :(real)),
                            (:(atanh),  "(1-f^2)*D",         "f'",       :(real)),
                            (:(erfcx),  "D-2f*f'",       "-2f'/sqrt(π)", :(real)),
                            (:(dawson), "D+2f*f'",           "f'",       :(real)))
    L,R = Meta.parse(ODE), Meta.parse(RHS)
    @eval begin
        # depice before doing op
        $op(f::Fun{<:PiecewiseSpace}) = Fun(map($op, components(f)),PiecewiseSpace)

        # We remove the MappedSpace
        # function $op{MS<:MappedSpace}(f::Fun{MS})
        #     g=exp(Fun(f.coefficients,space(f).space))
        #     Fun(g.coefficients,MappedSpace(domain(f),space(g)))
        # end
        function $op(fin::Fun)
            f=setcanonicaldomain(fin)  # removes possible issues with roots

            xmax,opfxmax,opmax=specialfunctionnormalizationpoint($op,$growth,f)
            # we will assume the result should be smooth on the domain
            # even if f is not
            # This supports Line/Rays
            D=Derivative(domain(f))
            B=Evaluation(domainspace(D),xmax)
            u=\([B, $L], [opfxmax, $R]; tolerance=eps(cfstype(fin))*opmax)

            setdomain(u,domain(fin))
        end
    end
end

Base.:(^)(::Irrational{:ℯ}, f::Fun) = exp(f)

function specialfunctionnormalizationpoint2(op, growth, f, T = cfstype(f))
    xmin, xmax, opfxmin, opfxmax, opmax = _specialfunctionnormalizationpoint(op,growth,f)
    while opmax≤10eps(T) || abs(f(xmin)-f(xmax))≤10eps(T)
        xmin,xmax = rand(domain(f)),rand(domain(f))
        opfxmin,opfxmax = op(f(xmin)),op(f(xmax))
        opmax = maximum(abs,(opfxmin,opfxmax))
    end
    xmin, xmax, opfxmin, opfxmax, opmax
end

for (op,ODE,RHS,growth) in ((:(erf),"f'*D^2+(2f*f'^2-f'')*D","0",:(imag)),
                            (:(erfi),"f'*D^2-(2f*f'^2+f'')*D","0",:(real)),
                            (:(sin),"f'*D^2-f''*D+f'^3","0",:(imag)),
                            (:(cos),"f'*D^2-f''*D+f'^3","0",:(imag)),
                            (:(sinh),"f'*D^2-f''*D-f'^3","0",:(real)),
                            (:(cosh),"f'*D^2-f''*D-f'^3","0",:(real)),
                            (:(airyai),"f'*D^2-f''*D-f*f'^3","0",:(imag)),
                            (:(airybi),"f'*D^2-f''*D-f*f'^3","0",:(imag)),
                            (:(airyaiprime),"f'*D^2-f''*D-f*f'^3","airyai(f)*f'^3",:(imag)),
                            (:(airybiprime),"f'*D^2-f''*D-f*f'^3","airybi(f)*f'^3",:(imag)))
    L,R = Meta.parse(ODE),Meta.parse(RHS)
    @eval begin
        function $op(fin::Fun)
            T = cfstype(fin)
            f=setcanonicaldomain(fin)
            xmin, xmax, opfxmin, opfxmax, opmax = specialfunctionnormalizationpoint2($op, $growth, f, T)
            S = space(f)
            B=[Evaluation(S,xmin), Evaluation(S,xmax)]
            D=Derivative(S)
            u=\([B;$L], [opfxmin;opfxmax;$R]; tolerance=10eps(T)*opmax)

            setdomain(u,domain(fin))
        end
    end
end

erfc(f::Fun) = 1-erf(f)


exp2(f::Fun) = exp(log(2)*f)
exp10(f::Fun) = exp(log(10)*f)
log2(f::Fun) = log(f)/log(2)
log10(f::Fun) = log(f)/log(10)

##TODO: the spacepromotion doesn't work for tan/tanh for a domain including zeros of cos/cosh inside.
tan(f::Fun) = sin(f)/cos(f) #This is inaccurate, but allows space promotion via division.
tanh(f::Fun) = sinh(f)/cosh(f) #This is inaccurate, but allows space promotion via division.

for (op,oprecip,opinv,opinvrecip) in ((:(sin),:(csc),:(asin),:(acsc)),
                                      (:(cos),:(sec),:(acos),:(asec)),
                                      (:(tan),:(cot),:(atan),:(acot)),
                                      (:(sinh),:(csch),:(asinh),:(acsch)),
                                      (:(cosh),:(sech),:(acosh),:(asech)),
                                      (:(tanh),:(coth),:(atanh),:(acoth)))
    @eval begin
        $oprecip(f::Fun) = 1/$op(f)
        $opinvrecip(f::Fun) = $opinv(1/f)
    end
end

rad2deg(f::Fun) = 180/π*f
deg2rad(f::Fun) = π/180*f

for (op,opd,opinv,opinvd) in ((:(sin),:(sind),:(asin),:(asind)),
                              (:(cos),:(cosd),:(acos),:(acosd)),
                              (:(tan),:(tand),:(atan),:(atand)),
                              (:(sec),:(secd),:(asec),:(asecd)),
                              (:(csc),:(cscd),:(acsc),:(acscd)),
                              (:(cot),:(cotd),:(acot),:(acotd)))
    @eval begin
        $opd(f::Fun) = $op(deg2rad(f))
        $opinvd(f::Fun) = rad2deg($opinv(f))
    end
end

#Won't get the zeros exactly 0 anyway so at least this way the length is smaller.
sinpi(f::Fun) = sin(π*f)
cospi(f::Fun) = cos(π*f)

function airy(k::Number,f::Fun)
    if k == 0
        airyai(f)
    elseif k == 1
        airyaiprime(f)
    elseif k == 2
        airybi(f)
    elseif k == 3
        airybiprime(f)
    else
        error("invalid argument")
    end
end

besselh(ν,k::Integer,f::Fun) = k == 1 ? hankelh1(ν,f) : k == 2 ? hankelh2(ν,f) : throw(Base.Math.AmosException(1))

for jy in (:j, :y)
    bjy = Symbol(:bessel, jy)
    for ν in (0,1)
        bjynu = Symbol(bjy, ν)
        @eval SpecialFunctions.$bjynu(f::Fun) = $bjy($ν,f)
    end
end

## Miscellaneous
for op in (:(expm1),:(log1p),:(lfact),:(sinc),:(cosc),
           :(erfinv),:(erfcinv),:(beta),:(lbeta),
           :(eta),:(zeta),:(gamma),:(lgamma),
           :(polygamma),:(invdigamma),:(digamma),:(trigamma))
    @eval begin
        $op(f::Fun) = Fun($op ∘ f,domain(f))
    end
end


## <,≤,>,≥

for op in (:<,)
    @eval begin
        function $op(f::Fun,c::Number)
            if length(roots(f-c))==0
                $op(first(f),c)
            else
                false
            end
        end
        function $op(c::Number,f::Fun)
            if length(roots(f-c))==0
                $op(c,first(f))
            else
                false
            end
        end
    end
end



for op in (:(<=),)
    @eval begin
        function $op(f::Fun,c::Number)
            rts=roots(f-c)
            if length(rts)==0
                $op(first(f),c)
            elseif length(rts)==1
                if isapprox(rts[1],leftendpoint(domain(f))) || isapprox(rts[1],rightendpoint(domain(f)))
                    $op(f(fromcanonical(f,0.)),c)
                else
                    error("Implement for mid roots")
                end
            elseif length(rts)==2
                if isapprox(rts[1],leftendpoint(domain(f))) && isapprox(rts[2],rightendpoint(domain(f)))
                    $op(f(fromcanonical(f,0.)),c)
                else
                    error("Implement for mid roots")
                end
            else
                error("Implement for mid roots")
            end
        end
        function $op(c::Number,f::Fun)
            rts=sort(roots(f-c))
            if length(rts)==0
                $op(c,first(f))
            elseif length(rts)==1
                if isapprox(rts[1],leftendpoint(domain(f))) || isapprox(rts[1],leftendpoint(domain(f)))
                    $op(c,f(fromcanonical(f,0.)))
                else
                    error("Implement for mid roots")
                end
            elseif length(rts)==2
                if isapprox(rts[1],leftendpoint(domain(f))) && isapprox(rts[2],leftendpoint(domain(f)))
                    $op(c,f(fromcanonical(f,0.)))
                else
                    error("Implement for mid roots")
                end
            else
                error("Implement for mid roots")
            end
        end
    end
end

/(c::Number,f::Fun{<:PiecewiseSpace}) = Fun(map(f->c/f,components(f)),PiecewiseSpace)
^(f::Fun{<:PiecewiseSpace},c::Integer) = Fun(map(f->f^c,components(f)),PiecewiseSpace)
^(f::Fun{<:PiecewiseSpace},c::Number) = Fun(map(f->f^c,components(f)),PiecewiseSpace)

for OP in (:abs,:sign,:log,:angle)
    @eval begin
        $OP(f::Fun{<:PiecewiseSpace{<:Any,<:Any,<:Real},<:Real}) =
            Fun(map($OP,components(f)),PiecewiseSpace)
        $OP(f::Fun{<:PiecewiseSpace{<:Any,<:Domain{<:Number}}}) =
            Fun(map($OP,components(f)),PiecewiseSpace)
    end
end

## Special Multiplication
for f in (:+, :-, :exp, :sin, :cos)
    @eval $f(M::Multiplication) = Multiplication($f(M.f), domainspace(M))
end

for f in (:+, :-, :*, :/, :\)
    @eval begin
        $f(M::Multiplication, c::Number) = Multiplication($f(M.f, c), domainspace(M))
        $f(c::Number, M::Multiplication) = Multiplication($f(c, M.f), domainspace(M))
    end
end

## ConstantSpace and PointSpace default overrides

for SP in (:ConstantSpace,:PointSpace)
    for OP in (:abs,:sign,:exp,:sqrt,:angle)
        @eval begin
            $OP(z::Fun{<:$SP,<:Complex}) = Fun(space(z),map($OP, coefficients(z)))
            $OP(z::Fun{<:$SP,<:Real}) = Fun(space(z),map($OP, coefficients(z)))
            $OP(z::Fun{<:$SP}) = Fun(space(z),map($OP, coefficients(z)))
        end
    end

    # we need to pad coefficients since 0^0 == 1
    for OP in (:^,)
        @eval begin
            function $OP(z::Fun{<:$SP},k::Integer)
                k ≠ 0 && return Fun(space(z),$OP.(coefficients(z),k))
                Fun(space(z),$OP.(pad(coefficients(z),dimension(space(z))),k))
            end
            function $OP(z::Fun{<:$SP},k::Number)
                k ≠ 0 && return Fun(space(z),$OP.(coefficients(z),k))
                Fun(space(z),$OP.(pad(coefficients(z),dimension(space(z))),k))
            end
        end
    end
end

for OP in (:<,:(Base.isless),:(<=))
    @eval begin
        $OP(a::Fun{<:ConstantSpace},b::Fun{<:ConstantSpace}) =
            $OP(strictconvert(Number,a), strictconvert(Number, b))
        $OP(a::Fun{<:ConstantSpace},b::Number) = $OP(strictconvert(Number,a),b)
        $OP(a::Number,b::Fun{<:ConstantSpace}) = $OP(a,strictconvert(Number,b))
    end
end

for OP in (:(Base.max),:(Base.min))
    @eval begin
        $OP(a::Fun{<:ConstantSpace,<:Real},b::Fun{<:ConstantSpace,<:Real}) =
            Fun($OP(Number(a),Number(b)),space(a) ∪ space(b))
        $OP(a::Fun{<:ConstantSpace,<:Real},b::Real) =
            Fun($OP(Number(a),b),space(a))
        $OP(a::Real,b::Fun{<:ConstantSpace,<:Real}) =
            Fun($OP(a,Number(b)),space(b))
    end
end

# from DualNumbers
for (funsym, exp) in Calculus.symbolic_derivatives_1arg()
    funsym == :abs && continue
    funsym == :sign && continue
    funsym == :exp && continue
    funsym == :sqrt && continue
    @eval begin
        $(funsym)(z::Fun{<:ConstantSpace,<:Real}) =
            Fun($(funsym)(Number(z)),space(z))
        $(funsym)(z::Fun{<:ConstantSpace,<:Complex}) =
            Fun($(funsym)(Number(z)),space(z))
        $(funsym)(z::Fun{<:ConstantSpace}) =
            Fun($(funsym)(Number(z)),space(z))
    end
end

# Other special functions
for f in [:logabsgamma]
    @eval function $f(z::Fun{<:ConstantSpace, <:Real})
        t = $f(Number(z))
        Fun(t[1], space(z)), t[2]
    end
end
function loggamma(z::Fun{<:ConstantSpace})
    t = loggamma(Number(z))
    Fun(t, space(z))
end
for f in [:gamma, :loggamma]
    @eval begin
        function $f(a, z::Fun{<:ConstantSpace})
            t = $f(a, Number(z))
            Fun(t, space(z))
        end
    end
end

for f in [:besselj, :besselk, :besselkx, :bessely, :besseli,
            :hankelh1x, :hankelh2x, :hankelh1, :hankelh2]
    @eval $f(nu, x::Fun{<:ConstantSpace}) = Fun($f(nu, Number(x)), space(x))
end

# Roots

for op in (:(argmax),:(argmin))
    @eval begin
        function $op(f::Fun{<:RealSpace,<:Real})
            # need to check for zero as extremal_args is not defined otherwise
            d = domain(f)
            T = eltype(d)
            iszero(f) && return leftendpoint(domain(f))
            # the following avoids warning when differentiate(f)==0
            pts = extremal_args(f)
            # the extra real avoids issues with complex round-off
            v = map(real∘f, pts)::Vector
            x = pts[strictconvert(Int, $op(v))]
            strictconvert(T, x)
        end

        function $op(f::Fun)
            # need to check for zero as extremal_args is not defined otherwise
            d = domain(f)
            T = eltype(d)
            iszero(f) && return leftendpoint(domain(f))
            # the following avoids warning when differentiate(f)==0
            pts = extremal_args(f)
            fp = map(f, pts)
            @assert norm(imag(fp))<100eps()
            v = real(fp)::Vector
            x = pts[strictconvert(Int, $op(v))]
            strictconvert(T, x)
        end
    end
end

if VERSION < v"1.7"
    _maybemap(rf, f, pts) = rf(map(f, pts))
else
    _maybemap(rf, f, pts) = rf(f, pts)
end

for op in (:(findmax),:(findmin))
    @eval begin
        function $op(f::Fun{<:RealSpace,<:Real})
            # the following avoids warning when differentiate(f)==0
            pts = extremal_args(f)
            ext,ind = _maybemap($op, f, pts)
    	    ext,pts[ind]
        end
    end
end

extremal_args(f::Fun{<:PiecewiseSpace}) = cat(1,[extremal_args(fp) for fp in components(f)], dims=1)

function extremal_args(f::Fun)
    d = domain(f)
    T = prectype(d)
    dab = strictconvert(Vector{T}, collect(components(∂(domain(f)))))
    if ncoefficients(f) > 2 #TODO this is only relevant for Polynomial bases
        r = roots(differentiate(f))
        if !isempty(r)
            append!(dab, r)
        end
    end
    return dab
end

for op in (:(maximum),:(minimum),:(extrema))
    @eval function $op(f::Fun{<:RealSpace,<:Real})
        pts = iszero(f') ? [leftendpoint(domain(f))] : extremal_args(f)
        _maybemap($op, f, pts)
    end
end


for op in (:(maximum),:(minimum))
    @eval begin
        function $op(::typeof(abs), f::Fun{<:RealSpace,<:Real})
            pts = iszero(f') ? [leftendpoint(domain(f))] : extremal_args(f)
            _maybemap($op, f, pts)
        end
        function $op(::typeof(abs), f::Fun)
            # complex spaces/types can have different extrema
            pts = extremal_args(abs(f))
            _maybemap($op, f, pts)
        end
        $op(f::Fun{PiecewiseSpace{<:Any,<:UnionDomain,<:Real},<:Real}) =
            $op(map($op,components(f)))
        $op(::typeof(abs), f::Fun{PiecewiseSpace{<:Any,<:UnionDomain,<:Real},<:Real}) =
            $op(abs, map(g -> $op(abs, g),components(f)))
    end
end


extrema(f::Fun{PiecewiseSpace{<:Any,<:UnionDomain,<:Real},<:Real}) =
    mapreduce(extrema,(x,y)->extrema([x...;y...]),components(f))

function companion_matrix(c::Vector{T}) where T
    n=length(c)-1

    A=zeros(T,n,n)

    for k=1:n
        A[k,end]=-c[k]/c[end]
    end

    for k=2:n
        A[k,k-1]=one(T)
    end

    return A
end

function complexroots end

complexroots(cfs::Vector{<:Union{Float64,ComplexF64}}) =
    hesseneigvals(companion_matrix(chop(cfs,10eps())))

complexroots(neg::Vector, pos::Vector) =
    complexroots([reverse(chop(neg,10eps()), dims=1); pos])

function roots(f::Fun{<:PiecewiseSpace})
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

roots(f::Fun{<:PointSpace}) = space(f).points[values(f) .== 0]

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


#function roots(f::Fun)
#    irts=map(real,filter!(x->abs(x)<=1.+10eps(),filter!#(isreal,complexroots(f.coefficients))))
#
#    map!(x->x>1. ? 1. : x,irts)
#    map!(x->x<-1. ? -1. : x,irts)
#
#    if length(irts)==0
#        Float64[]
#    else
#        fromcanonical(f,irts)
#    end
#end

function roots(f::Fun)
    f2=Fun(f,domain(f)) # default is to convert to Chebyshev/Fourier
    if space(f2)==space(f)
        error("roots not implemented for ", typeof(f))
    else
        roots(f2)
    end
end

#
# These formulæ, appearing in Eq. (2.5) of:
#
# A.-K. Kassam and L. N. Trefethen, Fourth-order time-stepping for stiff PDEs, SIAM J. Sci. Comput., 26:1214--1233, 2005,
#
# are derived to implement ETDRK4 in double precision without numerical instability from cancellation.
#

expα_asy(x) = (exp(x)*(4-3x+x^2)-4-x)/x^3
expβ_asy(x) = (exp(x)*(x-2)+x+2)/x^3
expγ_asy(x) = (exp(x)*(4-x)-4-3x-x^2)/x^3

# TODO: General types

expα_taylor(x::Union{Float64,ComplexF64}) = @evalpoly(x,1/6,1/6,3/40,1/45,5/1008,1/1120,7/51840,1/56700,1/492800,1/4790016,11/566092800,1/605404800,13/100590336000,1/106748928000,1/1580833013760,1/25009272288000,17/7155594141696000,1/7508956815360000)
expβ_taylor(x::Union{Float64,ComplexF64}) = @evalpoly(x,1/6,1/12,1/40,1/180,1/1008,1/6720,1/51840,1/453600,1/4435200,1/47900160,1/566092800,1/7264857600,1/100590336000,1/1494484992000,1/23712495206400,1/400148356608000,1/7155594141696000,1/135161222676480000)
expγ_taylor(x::Union{Float64,ComplexF64}) = @evalpoly(x,1/6,0/1,-1/120,-1/360,-1/1680,-1/10080,-1/72576,-1/604800,-1/5702400,-1/59875200,-1/691891200,-1/8717829120,-1/118879488000,-1/1743565824000,-1/27360571392000,-1/457312407552000,-1/8109673360588800)

expα(x::Float64) = abs(x) < 17/16 ? expα_taylor(x) : expα_asy(x)
expβ(x::Float64) = abs(x) < 19/16 ? expβ_taylor(x) : expβ_asy(x)
expγ(x::Float64) = abs(x) < 15/16 ? expγ_taylor(x) : expγ_asy(x)

expα(x::ComplexF64) = abs2(x) < (17/16)^2 ? expα_taylor(x) : expα_asy(x)
expβ(x::ComplexF64) = abs2(x) < (19/16)^2 ? expβ_taylor(x) : expβ_asy(x)
expγ(x::ComplexF64) = abs2(x) < (15/16)^2 ? expγ_taylor(x) : expγ_asy(x)

expα(x) = expα_asy(x)
expβ(x) = expβ_asy(x)
expγ(x) = expγ_asy(x)


for f in (:(exp),:(expm1),:expα,:expβ,:expγ)
    @eval $f(op::Operator) = OperatorFunction(op,$f)
end

