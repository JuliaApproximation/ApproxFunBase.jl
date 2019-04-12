## abs
splitmap(g,d::Domain,pts) = Fun(g,split(d , pts))

function split(d::IntervalOrSegment, pts)
    a,b = endpoints(d)
    isendpoint = true
    for p in pts
        if !(p ≈ a) && !(p ≈ b)
            isendpoint = false
            break
        end
    end
    isendpoint && return d

    @assert all(in.(pts, Ref(d)))
    PiecewiseSegment(sort!(union(endpoints(d), pts)))
end

function split(d::PiecewiseSegment, pts)
    @assert all(in.(pts, Ref(d)))
    PiecewiseSegment(sort!(union(d.points, pts)))
end

split(d::SegmentDomain, pts) = d


function splitatroots(f::Fun)
    d=domain(f)
    pts=union(roots(f)) # union removes multiplicities
    splitmap(x->f(x),d,pts)
end

function abs(f::Fun{S,T}) where {S<:RealUnivariateSpace,T<:Real}
    d=domain(f)
    pts = iszero(f) ? T[] : roots(f)
    splitmap(x->abs(f(x)),d,pts)
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

for OP in (:sign,:angle)
    @eval function $OP(f::Fun{S,T}) where {S<:RealUnivariateSpace,T<:Real}
        d=domain(f)

        pts = iszero(f) ? T[] : roots(f)

        if isempty(pts)
            $OP(first(f))*one(f)
        else
            d = split(d , pts)
            midpts = midpoints(d)
            Fun(UnionDomain(components(d)), $OP.(f.(midpts)))
        end
    end
end

for op in (:(max),:(min))
    @eval begin
        function $op(f::Fun{S,T1},g::Fun{V,T2}) where {S<:RealUnivariateSpace,V<:RealUnivariateSpace,T1<:Real,T2<:Real}
            h=f-g
            d=domain(h)
            pts=iszero(h) ? cfstype(h)[] : roots(h)
            splitmap(x->$op(f(x),g(x)),d,pts)
        end
        $op(f::Fun{S,T},g::Real) where {S<:RealUnivariateSpace,T<:Real} = $op(f,Fun(g,domain(f)))
        $op(f::Real,g::Fun{S,T}) where {S<:RealUnivariateSpace,T<:Real} = $op(Fun(f,domain(g)),g)
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
function ^(f::Fun{S,T},β) where {S,T}
    A=Derivative()-β*differentiate(f)/f
    B=Evaluation(leftendpoint(domain(f)))
    [B;A]\[first(f)^β;0]
end

sqrt(f::Fun{S,T}) where {S,T} = f^0.5
cbrt(f::Fun{S,T}) where {S,T} = f^(1/3)

## We use \ as the Fun constructor might miss isolated features

## First order functions


log(f::Fun) = cumsum(differentiate(f)/f)+log(first(f))

acos(f::Fun)=cumsum(-f'/sqrt(1-f^2))+acos(first(f))
asin(f::Fun)=cumsum(f'/sqrt(1-f^2))+asin(first(f))

## Second order functions
sin(f::Fun{S,T}) where {S<:RealSpace,T<:Real} = imag(exp(im*f))
cos(f::Fun{S,T}) where {S<:RealSpace,T<:Real} = real(exp(im*f))

atan(f::Fun)=cumsum(f'/(1+f^2))+atan(first(f))


# this is used to find a point in which to impose a boundary
# condition in calculating secial functions
function specialfunctionnormalizationpoint(op,growth,f)
    g=chop(growth(f),eps(cfstype(f)))
    xmin = isempty(g.coefficients) ? leftendpoint(domain(g)) : argmin(g)
    xmax = isempty(g.coefficients) ? rightendpoint(domain(g)) : argmax(g)
    opfxmin,opfxmax = op(f(xmin)),op(f(xmax))
    opmax = maximum(abs,(opfxmin,opfxmax))
    if abs(opfxmin) == opmax xmax,opfxmax = xmin,opfxmin end
    xmax,opfxmax,opmax
end

# ODE gives the first order ODE a special function op satisfies,
# RHS is the right hand side
# growth says what to use to choose a good point to impose an initial condition
for (op,ODE,RHS,growth) in ((:(exp),"D-f'","0",:(real)),
                            (:(asinh),"sqrt(f^2+1)*D","f'",:(real)),
                            (:(acosh),"sqrt(f^2-1)*D","f'",:(real)),
                            (:(atanh),"(1-f^2)*D","f'",:(real)),
                            (:(erfcx),"D-2f*f'","-2f'/sqrt(π)",:(real)),
                            (:(dawson),"D+2f*f'","f'",:(real)))
    L,R = Meta.parse(ODE),Meta.parse(RHS)
    @eval begin
        # depice before doing op
        $op(f::Fun{<:PiecewiseSpace}) = Fun(map(f->$op(f),components(f)),PiecewiseSpace)

        # We remove the MappedSpace
        # function $op{MS<:MappedSpace}(f::Fun{MS})
        #     g=exp(Fun(f.coefficients,space(f).space))
        #     Fun(g.coefficients,MappedSpace(domain(f),space(g)))
        # end
        function $op(fin::Fun{S,T}) where {S,T}
            f=setcanonicaldomain(fin)  # removes possible issues with roots

            xmax,opfxmax,opmax=specialfunctionnormalizationpoint($op,$growth,f)
            # we will assume the result should be smooth on the domain
            # even if f is not
            # This supports Line/Rays
            D=Derivative(domain(f))
            B=Evaluation(domainspace(D),xmax)
            u=\([B,eval($L)],Any[opfxmax,eval($R)];tolerance=eps(T)*opmax)

            setdomain(u,domain(fin))
        end
    end
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
        function $op(fin::Fun{S,T}) where {S,T}
            f=setcanonicaldomain(fin)

            g=chop($growth(f),eps(T))
            xmin = isempty(g.coefficients) ? leftendpoint(domain(g)) : argmin(g)
            xmax = isempty(g.coefficients) ? rightendpoint(domain(g)) : argmax(g)
            opfxmin,opfxmax = $op(f(xmin)),$op(f(xmax))
            opmax = maximum(abs,(opfxmin,opfxmax))
            while opmax≤10eps(T) || abs(f(xmin)-f(xmax))≤10eps(T)
                xmin,xmax = rand(domain(f)),rand(domain(f))
                opfxmin,opfxmax = $op(f(xmin)),$op(f(xmax))
                opmax = maximum(abs,(opfxmin,opfxmax))
            end
            D=Derivative(space(f))
            B=[Evaluation(space(f),xmin),Evaluation(space(f),xmax)]
            u=\([B;eval($L)],[opfxmin;opfxmax;eval($R)];tolerance=10eps(T)*opmax)

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

rad2deg(f::Fun) = 180*f/π
deg2rad(f::Fun) = π*f/180

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

for jy in ("j","y"), ν in (0,1)
    bjy = Symbol(string("bessel",jy))
    bjynu = Meta.parse(string("SpecialFunctions.bessel",jy,ν))
    @eval begin
        $bjynu(f::Fun) = $bjy($ν,f)
    end
end

## Miscellaneous
for op in (:(expm1),:(log1p),:(lfact),:(sinc),:(cosc),
           :(erfinv),:(erfcinv),:(beta),:(lbeta),
           :(eta),:(zeta),:(gamma),:(lgamma),
           :(polygamma),:(invdigamma),:(digamma),:(trigamma))
    @eval begin
        $op(f::Fun{S,T}) where {S,T}=Fun($op ∘ f,domain(f))
    end
end


## <,≤,>,≥

for op in (:<,:>)
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



for op in (:(<=),:(>=))
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

/(c::Number,f::Fun{S}) where {S<:PiecewiseSpace} = Fun(map(f->c/f,components(f)),PiecewiseSpace)
^(f::Fun{S},c::Integer) where {S<:PiecewiseSpace} = Fun(map(f->f^c,components(f)),PiecewiseSpace)
^(f::Fun{S},c::Number) where {S<:PiecewiseSpace} = Fun(map(f->f^c,components(f)),PiecewiseSpace)

for OP in (:abs,:sign,:log,:angle)
    @eval begin
        $OP(f::Fun{<:PiecewiseSpace{<:Any,<:Any,<:Real},<:Real}) =
            Fun(map($OP,components(f)),PiecewiseSpace)
        $OP(f::Fun{<:PiecewiseSpace{<:Any,<:Domain1d}}) =
            Fun(map($OP,components(f)),PiecewiseSpace)
    end
end

## Special Multiplication
for f in (:+, :-, :*, :exp, :sin, :cos)
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
            $OP(z::Fun{<:$SP,<:Complex}) = Fun(space(z),$OP.(coefficients(z)))
            $OP(z::Fun{<:$SP,<:Real}) = Fun(space(z),$OP.(coefficients(z)))
            $OP(z::Fun{<:$SP}) = Fun(space(z),$OP.(coefficients(z)))
        end
    end

    # we need to pad coefficients since 0^0 == 1
    for OP in (:^,)
        @eval begin
            function $OP(z::Fun{<:$SP},k::Integer)
                k ≠ 0 && return Fun(space(z),$OP.(coefficients(z),k))
                Fun(space(z),$OP.(pad(coefficients(z),dimension(space(z))),k))
            end
            function $OP(z::Fun{<:$SP},k::Number)
                k ≠ 0 && return Fun(space(z),$OP.(coefficients(z),k))
                Fun(space(z),$OP.(pad(coefficients(z),dimension(space(z))),k))
            end
        end
    end
end

for OP in (:<,:(Base.isless),:(<=),:>,:(>=))
    @eval begin
        $OP(a::Fun{CS},b::Fun{CS}) where {CS<:ConstantSpace} = $OP(convert(Number,a),Number(b))
        $OP(a::Fun{CS},b::Number) where {CS<:ConstantSpace} = $OP(convert(Number,a),b)
        $OP(a::Number,b::Fun{CS}) where {CS<:ConstantSpace} = $OP(a,convert(Number,b))
    end
end

for OP in (:(Base.max),:(Base.min))
    @eval begin
        $OP(a::Fun{CS1,T},b::Fun{CS2,V}) where {CS1<:ConstantSpace,CS2<:ConstantSpace,T<:Real,V<:Real} =
            Fun($OP(Number(a),Number(b)),space(a) ∪ space(b))
        $OP(a::Fun{CS,T},b::Real) where {CS<:ConstantSpace,T<:Real} =
            Fun($OP(Number(a),b),space(a))
        $OP(a::Real,b::Fun{CS,T}) where {CS<:ConstantSpace,T<:Real} =
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
        $(funsym)(z::Fun{CS,T}) where {CS<:ConstantSpace,T<:Real} =
            Fun($(funsym)(Number(z)),space(z))
        $(funsym)(z::Fun{CS,T}) where {CS<:ConstantSpace,T<:Complex} =
            Fun($(funsym)(Number(z)),space(z))
        $(funsym)(z::Fun{CS}) where {CS<:ConstantSpace} =
            Fun($(funsym)(Number(z)),space(z))
    end
end

# Roots

for op in (:(argmax),:(argmin))
    @eval begin
        function $op(f::Fun{S,T}) where {S<:RealSpace,T<:Real}
            # need to check for zero as extremal_args is not defined otherwise
            iszero(f) && return leftendpoint(domain(f))
            # the following avoids warning when differentiate(f)==0
            pts = extremal_args(f)
            # the extra real avoids issues with complex round-off
            pts[$op(real(f.(pts)))]
        end

        function $op(f::Fun{S,T}) where {S,T}
            # need to check for zero as extremal_args is not defined otherwise
            iszero(f) && return leftendpoint(domain(f))
            # the following avoids warning when differentiate(f)==0
            pts = extremal_args(f)
            fp=f.(pts)
            @assert norm(imag(fp))<100eps()
            pts[$op(real(fp))]
        end
    end
end

for op in (:(findmax),:(findmin))
    @eval begin
        function $op(f::Fun{S,T}) where {S<:RealSpace,T<:Real}
            # the following avoids warning when differentiate(f)==0
            pts = extremal_args(f)
            ext,ind = $op(f.(pts))
	    ext,pts[ind]
        end
    end
end

extremal_args(f::Fun{S}) where {S<:PiecewiseSpace} = cat(1,[extremal_args(fp) for fp in components(f)]...)

function extremal_args(f::Fun)
    d = domain(f)

    dab = convert(Vector{Number}, collect(components(∂(domain(f)))))
    if ncoefficients(f) <=2 #TODO this is only relevant for Polynomial bases
        dab
    else
        [dab;roots(differentiate(f))]
    end
end

for op in (:(maximum),:(minimum),:(extrema))
    @eval function $op(f::Fun{S,T}) where {S<:RealSpace,T<:Real}
        pts = iszero(f') ? [leftendpoint(domain(f))] : extremal_args(f)

        $op(f.(pts))
    end
end


for op in (:(maximum),:(minimum))
    @eval begin
        function $op(::typeof(abs), f::Fun{S,T}) where {S<:RealSpace,T<:Real}
            pts = iszero(f') ? [leftendpoint(domain(f))] : extremal_args(f)
            $op(f.(pts))
        end
        function $op(::typeof(abs), f::Fun)
            # complex spaces/types can have different extrema
            pts = extremal_args(abs(f))
            $op(f.(pts))
        end
        $op(f::Fun{PiecewiseSpace{SV,DD,RR},T}) where {SV,DD<:UnionDomain,RR<:Real,T<:Real} =
            $op(map($op,components(f)))
        $op(::typeof(abs), f::Fun{PiecewiseSpace{SV,DD,RR},T}) where {SV,DD<:UnionDomain,RR<:Real,T<:Real} =
            $op(abs, map(g -> $op(abs, g),components(f)))
    end
end


extrema(f::Fun{PiecewiseSpace{SV,DD,RR},T}) where {SV,DD<:UnionDomain,RR<:Real,T<:Real} =
    mapreduce(extrema,(x,y)->extrema([x...;y...]),components(f))

function complexroots end

function roots(f::Fun{P}) where P<:PiecewiseSpace
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

roots(f::Fun{S,T}) where {S<:PointSpace,T} = space(f).points[values(f) .== 0]



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

