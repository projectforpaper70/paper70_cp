using JuMP
using IntervalArithmetic

# We vendor ConditionalJuMP (https://github.com/rdeits/ConditionalJuMP.jl/blob/e0c406077c0b07be76e02f72c3a7a7aa650df82f/src/ConditionalJuMP.jl)
# so that we can use JuMP >= 0.2.0

owner_model(x::JuMP.VariableRef)::Model = JuMP.owner_model(x)
function owner_model(x::JuMP.GenericAffExpr)::Union{Model,Nothing}
    if length(x.terms) == 0
        return nothing
    end
    return JuMP.owner_model(first(x.terms.keys))
end
function owner_model(
    xs::AbstractArray{T},
)::Model where {T<:Union{JuMP.VariableRef,JuMP.GenericAffExpr}}
    for x in xs
        m = owner_model(x)
        if m !== nothing
            return m
        end
    end
    return nothing
end

# Supplements constructors in https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/src/intervals/construction.jl.
#IntervalArithmetic.interval(x::JuMP.VariableRef) =IntervalArithmetic.interval(lower_bound(x), upper_bound(x))
function getvar_itv(x::JuMP.VariableRef)
    lb_var = lower_bound(x)
    up_var = upper_bound(x)
    @assert(
        lb_var <= up_var,
        "lb of Var: $lb_var, does not lower than: $up_var."
    )
    return IntervalArithmetic.interval(lb_var, up_var)
end
function IntervalArithmetic.interval(e::JuMP.GenericAffExpr)
    result = IntervalArithmetic.interval(e.constant)
    for (var, coef) in e.terms
        itv_coef = IntervalArithmetic.interval(coef)
        itv_var = getvar_itv(var)
        result += itv_coef*itv_var
    end
    return result
end

lower_bound(x::Number) = x
upper_bound(x::Number) = x
lower_bound(x::JuMP.VariableRef) = JuMP.lower_bound(x)
upper_bound(x::JuMP.VariableRef) = JuMP.upper_bound(x)
lower_bound(e::JuMP.GenericAffExpr) = lower_bound(IntervalArithmetic.interval(e))
upper_bound(e::JuMP.GenericAffExpr) = upper_bound(IntervalArithmetic.interval(e))
lower_bound(i::Interval) = inf(i)
upper_bound(i::Interval) = sup(i)