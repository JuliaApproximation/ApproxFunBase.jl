module ApproxFunBaseDualNumbersExt

using ApproxFunBase
using DualNumbers
import ApproxFunBase: eps

eps(::Type{Dual{Complex{T}}}) where {T<:Real} = eps(real(T))
eps(z::Dual{Complex{T}}) where {T<:Real} = eps(abs(z))

end
