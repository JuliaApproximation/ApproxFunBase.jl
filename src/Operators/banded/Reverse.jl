

for TYP in (:ReverseOrientation,:Reverse)
    WRAP = Meta.parse(string(TYP)*"Wrapper")
    @eval begin
        abstract type $TYP{T} <: Operator{T} end

        struct $WRAP{T} <: Operator{T}
            op::Operator{T}
        end

        convert(::Type{Operator{T}},op::$TYP) where {T} = $TYP{T}()
        convert(::Type{Operator{T}},op::$WRAP) where {T} = $WRAP(Operator{T}(op.op))::Operator{T}

        @wrapper $WRAP
    end
end
