

for TYP in (:ReverseOrientation,:Reverse)
    WRAP = Symbol(TYP, :Wrapper)
    @eval begin
        abstract type $TYP{T} <: Operator{T} end

        struct $WRAP{OS,T} <: Operator{T}
            op::OS
        end

        $WRAP(op::Operator) = $WRAP{typeof(op),eltype(op)}(op)
        $TYP{T}(op::$TYP) where {T} = $TYP{T}()
        Operator{T}(op::$TYP) where {T} = $TYP{T}()
        $WRAP{OS,T}(op::$WRAP) where {OS,T} = $WRAP{OS,T}(strictconvert(OS,op))
        Operator{T}(op::$WRAP) where {T} = $WRAP(Operator{T}(op.op))::Operator{T}

        @wrapper $WRAP
    end
end
