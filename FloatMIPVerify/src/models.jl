using Serialization

"""
Supertype for types encoding the family of perturbations allowed.
"""
abstract type PerturbationFamily end

struct UnrestrictedPerturbationFamily <: PerturbationFamily end
Base.show(io::IO, pp::UnrestrictedPerturbationFamily) = print(io, "unrestricted")

abstract type RestrictedPerturbationFamily <: PerturbationFamily end

"""
For blurring perturbations, we currently allow colors to "bleed" across color channels -
that is, the value of the output of channel 1 can depend on the input to all channels.
(This is something that is worth reconsidering if we are working on color input).
"""
struct BlurringPerturbationFamily <: RestrictedPerturbationFamily
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurringPerturbationFamily) =
    print(io, filter(x -> !isspace(x), "blur-$(pp.blur_kernel_size)"))

struct LInfNormBoundedPerturbationFamily <: RestrictedPerturbationFamily
    norm_bound::Real

    function LInfNormBoundedPerturbationFamily(norm_bound::Real)
        @assert(norm_bound > 0, "Norm bound $(norm_bound) should be positive")
        return new(norm_bound)
    end
end
Base.show(io::IO, pp::LInfNormBoundedPerturbationFamily) =
    print(io, "linf-norm-bounded-$(pp.norm_bound)")

function get_model(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
)::Dict{Symbol,Any}
    # notice(
    #     MIPVerify.LOGGER,
    #     "Determining upper and lower bounds for the input to each non-linear unit.",
    # )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return merge(d_common, get_perturbation_specific_keys(nn, input, pp, m))
end

function get_model_check_robust(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
    radius::Real
)::Dict{Symbol,Any}
    # notice(
    #     MIPVerify.LOGGER,
    #     "Determining upper and lower bounds for the input to each non-linear unit.",
    # )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return merge(d_common, set_robustradius_checkrobust(nn, input, pp, m, radius))
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))

    # v_x0 is the input with the perturbation added也有人将其用于存储位数更少的8位定点量化网络的验证中
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_x0 - input, :Output => v_output)
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::BlurringPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_size = size(input)
    num_channels = size(input)[4]
    filter_size = (pp.blur_kernel_size..., num_channels, num_channels)

    v_f = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(filter_size))
    @constraint(m, sum(v_f) == num_channels)
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(input_size))
    @constraint(m, v_x0 .== input |> Conv2d(v_f))

    v_output = v_x0 |> nn

    return Dict(
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
    )
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    #println("the lenght of perturbation var is : ",length(v_e))
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - pp.norm_bound),
            upper_bound = min(1, input[i] + pp.norm_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end


function set_robustradius_checkrobust(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    m::Model,
    radius::Real,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    #添加jump部分约束
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - radius),
            upper_bound = min(1, input[i] + radius)
        ),
        input_range,
    )
    @constraint(m, v_x0 .<= input .+ radius)
    @constraint(m, v_x0 .>= input .- radius)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Output => v_output)

end

##添加sagemath输入约束
function get_perturbation_specific_keys_withsage(
    sgModel::PyObject,
    m::Model,
    nn_itv::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    input_size = size(input,1)
    # v_e is the perturbation added

    #首先添加sagemath部分的约束
    v_e_sage = sgModel.new_variable()
    v_e_itv = [v_e_sage[i] for i in 1:input_size]
    for i in 1:input_size
        # sgModel.set_min(v_e_itv[i],-pp.norm_bound)
        sgModel.set_min(v_e_itv[i],0)
        sgModel.set_max(v_e_itv[i],pp.norm_bound)
    end

    

    v_x0_sage = sgModel.new_variable()
    v_x0_itv = [v_x0_sage[i] for i in 1:input_size]
    for i in 1:input_size
        sgModel.set_min(v_x0_itv[i],max(0, input[i] - pp.norm_bound)),
        sgModel.set_max(v_x0_itv[i],min(1, input[i] + pp.norm_bound))
    end

    for i in 1:input_size
        sgModel.add_constraint(v_x0_itv[i] <= input[i]+v_e_itv[i])
        sgModel.add_constraint(v_x0_itv[i] >= input[i]-v_e_itv[i])
    end

    #添加jump部分约束
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    #println("the lenght of perturbation var is : ",length(v_e))
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - pp.norm_bound),
            upper_bound = min(1, input[i] + pp.norm_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output,v_output_itv=nn_itv(v_x0,v_x0_itv)

    return Dict(:PerturbedInput_itv=>v_x0_itv,:Perturbation_itv=>v_e_itv,:Output_itv=>v_output_itv,:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end


##添加sagemath输入约束
function set_robustradius_checkrobust_withsage(
    sgModel::PyObject,
    m::Model,
    nn_itv::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    radius::Real = 0.0001
)::Dict{Symbol,Any}

    radius = radius + 0.0001
    input_range = CartesianIndices(size(input))
    input_size = size(input,1)
    # v_e is the perturbation added

    #首先添加sagemath部分的约束
    # v_e_sage = sgModel.new_variable()
    # v_e_itv = [v_e_sage[i] for i in 1:input_size]
    # for i in 1:input_size
    #     # sgModel.set_min(v_e_itv[i],-pp.norm_bound)
    #     sgModel.set_min(v_e_itv[i],0)
    #     sgModel.set_max(v_e_itv[i],pp.norm_bound)
    # end

    

    v_x0_sage = sgModel.new_variable()
    v_x0_itv = [v_x0_sage[i] for i in 1:input_size]
    for i in 1:input_size
        sgModel.set_min(v_x0_itv[i],max(0, input[i] - pp.norm_bound)),
        sgModel.set_max(v_x0_itv[i],min(1, input[i] + pp.norm_bound))
    end

    for i in 1:input_size
        sgModel.add_constraint(v_x0_itv[i] <= input[i]+radius)
        sgModel.add_constraint(v_x0_itv[i] >= input[i]-radius)
    end

    #添加jump部分约束
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    #println("the lenght of perturbation var is : ",length(v_e))
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - radius),
            upper_bound = min(1, input[i] + radius)
        ),
        input_range,
    )
    @constraint(m, v_x0 .<= input .+ radius)
    @constraint(m, v_x0 .>= input .- radius)

    v_output,v_output_itv=nn_itv(v_x0,v_x0_itv)

    return Dict(:PerturbedInput_itv=>v_x0_itv,:Output_itv=>v_output_itv,:PerturbedInput => v_x0, :Output => v_output)
end


struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
end
