module ApproxFunBaseSpecialFunctionsExt

using ApproxFunBase
using SpecialFunctions
import Calculus

# we need to import all special functions to use Calculus.symbolic_derivatives_1arg
# we can't do importall Base as we replace some Base definitions
import SpecialFunctions: airy, besselh,
              lfact, beta, lbeta,
              eta, zeta, polygamma, logabsgamma, loggamma,
              besselj, bessely, besseli, besselk, besselkx,
              hankelh1, hankelh2, hankelh1x, hankelh2x,
              # functions from Calculus.symbolic_derivatives_1arg
              erf, erfinv, erfc, erfcinv, erfi, gamma, lgamma,
              digamma, invdigamma, trigamma,
              airyai, airybi, airyaiprime, airybiprime,
              besselj0, besselj1, bessely0, bessely1,
              erfcx, dawson

besselh(ν,k::Integer,f::Fun) = k == 1 ? hankelh1(ν,f) : k == 2 ? hankelh2(ν,f) : throw(Base.Math.AmosException(1))

for jy in (:j, :y)
    bjy = Symbol(:bessel, jy)
    for ν in (0,1)
        bjynu = Symbol(bjy, ν)
        @eval SpecialFunctions.$bjynu(f::Fun) = $bjy($ν,f)
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


for (funsym, exp) in Calculus.symbolic_derivatives_1arg()
    if isdefined(SpecialFunctions, funsym)
        @eval begin
            $(funsym)(z::Fun{<:ConstantSpace,<:Real}) =
                Fun($(funsym)(Number(z)),space(z))
            $(funsym)(z::Fun{<:ConstantSpace,<:Complex}) =
                Fun($(funsym)(Number(z)),space(z))
            $(funsym)(z::Fun{<:ConstantSpace}) =
                Fun($(funsym)(Number(z)),space(z))
        end
    end
end

for op in (:(lfact),:(erfinv),:(erfcinv),:(beta),:(lbeta),
           :(eta),:(zeta),:(gamma),:(lgamma),
           :(polygamma),:(invdigamma),:(digamma),:(trigamma))
    @eval begin
        $op(f::Fun) = Fun($op ∘ f,domain(f))
    end
end


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

erfc(f::Fun) = 1-erf(f)

for (op,ODE,RHS,growth) in ((:(erf),"f'*D^2+(2f*f'^2-f'')*D","0",:(imag)),
                            (:(erfi),"f'*D^2-(2f*f'^2+f'')*D","0",:(real)),
                            (:(airyai),"f'*D^2-f''*D-f*f'^3","0",:(imag)),
                            (:(airybi),"f'*D^2-f''*D-f*f'^3","0",:(imag)),
                            (:(airyaiprime),"f'*D^2-f''*D-f*f'^3","airyai(f)*f'^3",:(imag)),
                            (:(airybiprime),"f'*D^2-f''*D-f*f'^3","airybi(f)*f'^3",:(imag)))
    L,R = Meta.parse(ODE),Meta.parse(RHS)
    @eval begin
        function $op(fin::Fun)
            T = ApproxFunBase.cfstype(fin)
            f= ApproxFunBase.setcanonicaldomain(fin)
            xmin, xmax, opfxmin, opfxmax, opmax =
            	ApproxFunBase.specialfunctionnormalizationpoint2($op, $growth, f, T)
            S = space(f)
            B=[Evaluation(S,xmin), Evaluation(S,xmax)]
            D=Derivative(S)
            u=\([B;$L], [opfxmin;opfxmax;$R]; tolerance=10eps(T)*opmax)

            setdomain(u, domain(fin))
        end
    end
end

# ODE gives the first order ODE a special function op satisfies,
# RHS is the right hand side
# growth says what to use to choose a good point to impose an initial condition
for (op, ODE, RHS, growth) in (
                            (:(erfcx),  "D-2f*f'",       "-2f'/Base.sqrt(π)", :(real)),
                            (:(dawson), "D+2f*f'",           "f'",       :(real))
                            )
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
            f=ApproxFunBase.setcanonicaldomain(fin)  # removes possible issues with roots

            xmax, opfxmax, opmax =
            	ApproxFunBase.specialfunctionnormalizationpoint($op,$growth,f)
            # we will assume the result should be smooth on the domain
            # even if f is not
            # This supports Line/Rays
            D=Derivative(domain(f))
            B=Evaluation(domainspace(D),xmax)
            u=\([B, $L], [opfxmax, $R]; tolerance=eps(ApproxFunBase.cfstype(fin))*opmax)

            setdomain(u, domain(fin))
        end
    end
end

end
