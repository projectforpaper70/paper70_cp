export relu
using JuMP
using Memento
using MathOptInterface


function relu(m::P, x::T, x_itv::P, l::Real, u::Real)::PyObject where {P<:PyObject,T<:JuMPLinearType}
    # println("upper bound is :",u)
    # println("lower bound is :",l)
    if u < l
        # TODO (vtjeng): This check is in place in case of numerical error in the calculation of bounds.
        # See sample number 4872 (1-indexed) when verified on the lp0.4 network.
        Memento.warn(
            MIPVerify.LOGGER,
            "Inconsistent upper and lower bounds: u-l = $(u - l) is negative. Attempting to use interval arithmetic bounds instead ...",
        )
        #如果出错了就将u,l设置成最初的新建变量时候的上下界
        u = upper_bound(x)
        l = lower_bound(x)
    end

    # println("\n打印上界u和下界l:")
    # println("u:",u)
    # println("l:",l)
    # println("\n")

    if u <= 0
        # rectified value is always 0
        x_itv_rect = m.new_variable()
        m.add_constraint(x_itv_rect[0] == 0)
        # Manually set the bounds for x_rect so they can be used by downstream operations.
        m.set_min(x_itv_rect[0], l)
        m.set_max(x_itv_rect[0], 0)
        return zero(T),x_itv_rect[0]
    elseif u == l
        x_itv_rect = m.new_variable()
        m.add_constraint(x_itv_rect[0] == l)
        m.set_min(x_itv_rect[0], l)
        m.set_max(x_itv_rect[0], u)
        return one(T) * l,x_itv_rect[0]
    elseif u < l
        error(
            MIPVerify.LOGGER,
            "Inconsistent upper and lower bounds even after using only interval arithmetic: u-l = $(u - l) is negative",
        )
    elseif l >= 0
        # rectified value is always x
        # for x_itv it turn to a variable in range of x_itv
        x_itv_rect = m.new_variable()
        m.add_constraint(x_itv_rect[0] <= x_itv.var_up)
        m.add_constraint(x_itv_rect[0] >= x_itv.var_lo)
        # Manually set the bounds for x_rect so they can be used by downstream operations.
        m.set_min(x_itv_rect[0], l)
        m.set_max(x_itv_rect[0], u)
        return x,x_itv_rect[0]
    else
        #println("进入激活未知状态：")
        
        # since we know that u!=l, x is not constant, and thus x must have an associated model
        model = owner_model(x)
        x_rect = @variable(model)
        a = @variable(model, binary = true)

        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l) * (1 - a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u * a)
        @constraint(model, x_rect >= 0)

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        set_lower_bound(x_rect, l)
        set_upper_bound(x_rect, u)
        
        
        # since we know that u!=l, x is not constant, and thus x must have an associated model
        binary_phi = m.new_variable(binary=true)
        x_itv_rect  =  m.new_variable()
        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.

        # m.add_constraint(x_itv_rect[0] <= x_itv.var_up + (-l) * (1 - binary_phi[0]) )
        # m.add_constraint(x_itv_rect[0] >= x_itv.var_lo)
        # m.add_constraint(x_itv_rect[0] >= 0)
        # m.add_constraint(x_itv_rect[0] <= u*binary_phi[0])


        m.add_constraint(x_itv_rect[0] <= x_itv.var_up + 100*binary_phi[0] )
        m.add_constraint(x_itv_rect[0] >= x_itv.var_lo)
        m.add_constraint(x_itv_rect[0] >= 0)
        m.add_constraint(x_itv_rect[0] <= 100*(1-binary_phi[0]))

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        m.set_min(x_itv_rect[0], l)
        m.set_max(x_itv_rect[0], u)

        return x_rect,x_itv_rect[0]
    end
end


# #区间版本
# function relu(m::P, x::T)::PyObject where {T<:Interval_Var,P<:PyObject}
#     binary_phi = p.new_variable(binary=true)
#     Big_M =1000
#     z =  p.new_variable()
#     #设置第二层神经元变量
#     p.add_constraint(z[0] <= x.var_up + Big_M*binary_phi[0])
#     p.add_constraint(z[0] >= x.var_lo)
#     p.add_constraint(z[0] >= 0)
#     p.add_constraint(z[0] <= Big_M*(1-binary_phi[0]))
#     return z[0]
# end


# """
# $(SIGNATURES)
# Expresses a rectified-linearity constraint: output is constrained to be equal to
# `max(x, 0)`.
# """
function relu(
    m::PyObject,
    x::AbstractArray{T},
    x_itv::AbstractArray{PyObject};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
)::Tuple{Vector{T}, Vector{PyObject}} where {T<:JuMPLinearType}
    show_progress_bar::Bool =
        MIPVerify.LOGGER.levels[MIPVerify.LOGGER.level] > MIPVerify.LOGGER.levels["debug"]
    #println("the show_progress_bar is :",show_progress_bar)   
    if !show_progress_bar
        u = tight_upperbound.(x, nta = nta, cutoff = 0)
        l = lazy_tight_lowerbound.(x, u, nta = nta, cutoff = 0)
        return relu.(m, x, x_itv, l, u)
    else
        p1 = Progress(length(x), desc = "  Calculating upper bounds: ", enabled = isinteractive())
        u = map(x_i -> (next!(p1); tight_upperbound(x_i, nta = nta, cutoff = 0)), x)
        p2 = Progress(length(x), desc = "  Calculating lower bounds: ", enabled = isinteractive())
        l = map(v -> (next!(p2); lazy_tight_lowerbound(v..., nta = nta, cutoff = 0)), zip(x, u))

        reluinfo = ReLUInfo(l, u)
        Memento.info(MIPVerify.LOGGER, "$reluinfo")

        p3 = Progress(length(x), desc = "  Imposing relu constraint: ", enabled = isinteractive())

        # bound_tuple = zip(x,x_itv,l,u)
        # println("the size of bound_tuple is : ",size(bound_tuple))
        # size_bound_tuple = size(bound_tuple)
        x_r = []
        result_arr1 = Vector{T}(undef, length(x))
        result_arr2 = Vector{PyObject}(undef, length(x_itv))
        #return_x,return_xtv = nothing ,nothing
        for i in 1:size(x,1)
            next!(p3)
            result_arr1[i],result_arr2[i]=relu(m, x[i], x_itv[i],l[i], u[i] )
        end

        #println("测试,运行的是预期的relu激活函数")

        return result_arr1,result_arr2
    end
end