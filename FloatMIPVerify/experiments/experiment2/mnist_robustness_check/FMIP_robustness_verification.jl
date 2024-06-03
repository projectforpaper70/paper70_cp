if length(ARGS) < 7
    println("""
    Please provide the following parameters:
    1. data_type (Int) - e.g., 1 or 2
    2. data_name (String) - should be "MNIST", "mnist49", or "FASHIONMNIST"
    3. network_name (String) - should be "F16MNISTinput_77", "F16MNIST24", or "F16Fashion24"
    4. itv_network_name (String) - should be "F16MNISTinput_77itv", "F16MNIST24itv", or "F16Fashion784itv"
    5. min_data_num (Int) - e.g., 0
    6. max_data_num (Int) - e.g., 10000
    7. epsilon (Float64) - e.g., 0 0.05 0.1
    8. solver (optional, String) - "ppl" or "gurobi"
    """)
    exit(0)
end

try
    using PyCall
    sage_all = pyimport("sage.all")
catch
    # println("Failed to import sage.all")
end
using PyCall
sage_all=pyimport("sage.all")
using Random
using IntervalArithmetic
using Gurobi
using JuMP
using MIPVerify
using MIPVerify: relu, NNetFormat, evaluate_network, evaluate_network_withoutNorm, evaluate_network_multiple, info
using MIPVerify: prep_data_file, get_matrix_params, JuMPLinearType
using MAT
using Images

struct VerifyInfo
    index::Int
    time::Float64
    adv_found::Bool
    adv_not_found::Bool
    verified::Bool
    verified_res::Int
end

function fmp_robust_verify(ARGS)
    
    data_type = parse(Int, ARGS[1])
    data_name = ARGS[2]  # should be "MNIST" or "mnist49" or "FASHIONMNIST"
    network_name = ARGS[3]  # should be "F16MNISTinput_77" or "F16MNIST24" or "F16Fashion24"
    itv_network_name = ARGS[4]  # should be "F16MNISTinput_77itv" or "F16MNIST24itv" or "F16Fashion784itv"
    min_data_num = parse(Int, ARGS[5])
    max_data_num = parse(Int, ARGS[6])
    epsilon = parse(Float64, ARGS[7])

    # Default value for solver
    mip_solver = "gurobi"

    # Check if solver argument is provided
    if length(ARGS) >= 8
        solver_arg = ARGS[8]
        if solver_arg in ["gurobi", "ppl"]
            mip_solver = solver_arg
        else
            println("Unknown solver: $solver_arg. Using default solver: gurobi")
        end
    end

    println("Parameters received:")
    println("data_type: $data_type")
    println("data_name: $data_name")
    println("network_name: $network_name")
    println("itv_network_name: $itv_network_name")
    println("min_data_num: $min_data_num")
    println("max_data_num: $max_data_num")
    println("epsilon: $epsilon")
    println("solver: $mip_solver")

    misclassified = 0
    result = VerifyInfo[]
    round_error_class = Int[]

    if data_type == 1
        println("verification 7*7 input")
        if data_name == "mnist49"
            binary_file_path = "/home/aritifact/aritifact_for_cp24/FloatMIPVerify/resized_images/resized_mnist_images77_test.bin"
        end
        image_size = (7, 7)
        num_images = 10000
        images, labels = load_all_binary_images(binary_file_path, image_size, num_images)
        nn = MIPVerify.get_example_network_params(network_name)
        for index in min_data_num:max_data_num
            total_time = 0
            try
                total_time = @elapsed begin
                    sample_image = reshape(images[index], (1, 7, 7, 1))
                    sample_image16 = Float16.(sample_image)
                    sample_label = labels[index]
                    predicted_output = sample_image16 |> nn
                    println("index is: $index")
                    println(predicted_output)
                    predicted_output_index = argmax(predicted_output)
                    if predicted_output_index != sample_label + 1
                        println("Sample $index misclassified.")
                        println(predicted_output_index)
                        println(sample_label + 1)
                        misclassified += 1
                        #continue
                    end
                    println("--------verifying index:$index ---------")
                    @assert (predicted_output_index - 1) == sample_label

                    flat_layer = Flatten(4)
                    flat_img = flat_layer(sample_image16)
                    input_size = 49
                    @assert(
                        length(flat_img) == input_size, 
                        "the length of flat img: $(length(flat_img)) is not equal to the input size: $input_size"
                    )

                    ##约束求解部分
                    sgSolver = sage_all.MixedIntegerLinearProgram(solver=mip_solver)
                    model = Model(Gurobi.Optimizer)
                    set_optimizer_attribute(model, "OutputFlag", 0)
                    nn_itv = MIPVerify.get_example_network_params_withSageSolver(itv_network_name, sgSolver)
                    d1 = MIPVerify.set_robustradius_checkrobust_withsage(
                        sgSolver,
                        model,
                        nn_itv,
                        flat_img,
                        MIPVerify.LInfNormBoundedPerturbationFamily(0.15),
                        epsilon
                    )

                    ##读取MILP编码bound值
                    output_jump = d1[:Output]
                    output_var = sgSolver.new_variable()
                    output_itv = d1[:Output_itv]

                    for i in 1:size(d1[:Output_itv], 1)
                        sgSolver.add_constraint(output_var[i] <= output_itv[i].var_up)
                        sgSolver.add_constraint(output_var[i] >= output_itv[i].var_lo)
                    end

                    ##设置输出约束
                    label_index = 1:10
                    target_index = [i for i in label_index if i != predicted_output_index]

                    output_var_list = [output_var[i] for i in 1:10]
                    @assert(
                        all(item != predicted_output_index for item in target_index), 
                        "the length of flat img: $(target_index) should not equal to the : $(predicted_output_index)"
                    )

                    MIPVerify.set_max_indexes_withsage(sgSolver, output_jump, output_var_list, target_index, margin=floatmin(Float64))

                    @elapsed sgSolver.solve()
                    println("verified unknown\n")
                    solved_input = [convert(Float16, sgSolver.get_values(d1[:PerturbedInput_itv][i])) for i in 1:input_size]
                    perturbation_predicted_output = solved_input |> nn
                    perturbation_predicted_index = argmax(perturbation_predicted_output)
                    if perturbation_predicted_index == predicted_output_index
                        false_adv_res = VerifyInfo(index, total_time, false, false, false, 0)
                        push!(result, false_adv_res)
                    else
                        true_adv_res = VerifyInfo(index, total_time, true, false, true, -1)
                        push!(result, true_adv_res)
                    end
                end
            catch e
                println("verified true\n")
                verified_robust = VerifyInfo(index, total_time, false, true, true, 1)
                push!(result, verified_robust)
            end
        end

    elseif data_type == 2
        println("verification 28*28 input")
        mnist = MIPVerify.read_datasets(data_name)
        nn784 = MIPVerify.get_example_network_params(network_name)
        for index in min_data_num:max_data_num
            total_time = 0
            try
                total_time = @elapsed begin
                    sample_image = MIPVerify.get_image(mnist.test.images, index)
                    sample_label = MIPVerify.get_label(mnist.test.labels, index)
                    predicted_output = Float16.(sample_image) |> nn784
                    predicted_output_index = argmax(predicted_output)
                    if predicted_output_index != sample_label + 1
                        println("Sample $index misclassified.")
                        misclassified += 1
                        continue
                    end
                    println("--------verifying index:$index ---------")
                    @assert (predicted_output_index - 1) == sample_label

                    flat_layer = Flatten(4)
                    flat_img = flat_layer(sample_image)
                    input_size = 784
                    @assert(
                        length(flat_img) == input_size, 
                        "the length of flat img: $(length(flat_img)) is not equal to the input size: $input_size"
                    )

                    ##约束求解部分
                    sgSolver = sage_all.MixedIntegerLinearProgram(solver=mip_solver)
                    model = Model(Gurobi.Optimizer)
                    set_optimizer_attribute(model, "OutputFlag", 0)
                    nn_itv = MIPVerify.get_example_network_params_withSageSolver(itv_network_name, sgSolver)
                    d1 = MIPVerify.set_robustradius_checkrobust_withsage(
                        sgSolver,
                        model,
                        nn_itv,
                        flat_img,
                        MIPVerify.LInfNormBoundedPerturbationFamily(0.15),
                        epsilon
                    )

                    ##读取MILP编码bound值
                    output_jump = d1[:Output]
                    output_var = sgSolver.new_variable()
                    output_itv = d1[:Output_itv]

                    for i in 1:size(d1[:Output_itv], 1)
                        sgSolver.add_constraint(output_var[i] <= output_itv[i].var_up)
                        sgSolver.add_constraint(output_var[i] >= output_itv[i].var_lo)
                    end

                    ##设置输出约束
                    label_index = 1:10
                    target_index = [i for i in label_index if i != predicted_output_index]

                    output_var_list = [output_var[i] for i in 1:10]
                    @assert(
                        all(item != predicted_output_index for item in target_index), 
                        "the length of flat img: $(target_index) should not equal to the : $(predicted_output_index)"
                    )

                    MIPVerify.set_max_indexes_withsage(sgSolver, output_jump, output_var_list, target_index, margin=floatmin(Float64))

                    @elapsed sgSolver.solve()
                    println("verified unknown\n")
                    solved_input = [convert(Float16, sgSolver.get_values(d1[:PerturbedInput_itv][i])) for i in 1:input_size]
                    perturbation_predicted_output = solved_input |> nn784
                    perturbation_predicted_index = argmax(perturbation_predicted_output)
                    if perturbation_predicted_index == predicted_output_index
                        false_adv_res = VerifyInfo(index, total_time, false, false, false, 0)
                        push!(result, false_adv_res)
                    else
                        true_adv_res = VerifyInfo(index, total_time, true, false, true, -1)
                        push!(result, true_adv_res)
                    end
                end
            catch e
                println("verified true\n")
                verified_robust = VerifyInfo(index, total_time, false, true, true, 1)
                push!(result, verified_robust)
            end
        end
    end

    run_statistics = Dict(
        :result => result,
        :misclassified => misclassified,
        :round_error_class => round_error_class
    )
    return run_statistics
end

return_dict = fmp_robust_verify(ARGS)
res = return_dict[:result]

data_name = ARGS[2]
network_name = ARGS[3]
min_data_num = parse(Int, ARGS[5])
max_data_num = parse(Int, ARGS[6])
epsilon = parse(Float64, ARGS[7])

verify_unknown = [i.index for i in res if i.verified_res != 1]
verify_true = [i.index for i in res if i.verified && i.verified_res == 1]

total_num = max_data_num - min_data_num + 1 - return_dict[:misclassified]

println("total: ", total_num)
println("unknown: ", length(verify_unknown))
println("true: ", length(verify_true))

output_file = "./fp_verify_res/fpres_$(network_name)_$(data_name)_ep$(epsilon)_from_index$(min_data_num)to$(max_data_num).txt"

open(output_file, "w") do file
    println(file, "verify_unknown: ", join(verify_unknown, ", "))
    println(file, "verify_true: ", join(verify_true, ", "))
end
