if length(ARGS) < 6
    println("""
    Please provide the following parameters:
    1. data_type (Int) - e.g., 1 or 2
    2. data_name (String) - should be "MNIST", "mnist49", or "FASHIONMNIST"
    3. network_name (String) - should be "F16MNISTinput_77", "F16MNIST24", or "F16Fashion24"
    4. min_data_num (Int) - e.g., 0
    5. max_data_num (Int) - e.g., 10000
    6. epsilon (Float64) - e.g., 0 0.05 0.1
    """)
    exit(0)
end

using MIPVerify
using Gurobi
using JuMP
using Images
using Printf
function print_summary(d::Dict)
    # Helper function to print out output
    obj_val = JuMP.objective_value(d[:Model])
    solve_time = JuMP.solve_time(d[:Model])
    println("Objective Value: $(obj_val), Solve Time: $(@sprintf("%.2f", solve_time))")
end

function view_diff(diff::Array{<:Real,2})
    n = 1001
    colormap("RdBu", n)[ceil.(Int, (diff .+ 1) ./ 2 .* n)]
end


struct VerifyInfo
    index::Int
    time::Float64
    adv_found::Bool
    adv_not_found::Bool
    verified::Bool
    verified_res::Int
end


function mp_robust_verify()

    data_type = parse(Int, ARGS[1])
    data_name = ARGS[2]#should be "MNIST" or "mnist49" or "FASHIONMNIST"
    network_name = ARGS[3]#should be "F16MNISTinput_77" or "F16MNIST24" or "F16Fashion24"
    min_data_num = parse(Int, ARGS[4])# example like 0
    max_data_num = parse(Int, ARGS[5])# example like 10000
    epsilon = parse(Float64, ARGS[6]) # 0 ,0.05 ,0.1

    println("Parameters received:")
    println("data_type: $data_type")
    println("data_name: $data_name")
    println("network_name: $network_name")
    println("min_data_num: $min_data_num")
    println("max_data_num: $max_data_num")
    println("epsilon: $epsilon")

    if data_type == 1
        println("verification 7*7 input")
        misclassified = 0
        result = VerifyInfo[]
        round_error_class = Int[]
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
                ##参数读取部分代码
                sample_image = reshape(images[index], (1, 7, 7, 1))
                sample_image16 = Float16.(sample_image)
                sample_label = labels[index]
                predicted_output = sample_image16 |> nn
                #println(predicted_output)
                predicted_output_index = argmax(predicted_output)
                if predicted_output_index != sample_label + 1
                    println("Sample $index misclassified.")
                    misclassified = misclassified + 1
                    continue
                end
                println("--------verifying index:$index ---------")
                #println("the label : $sample_label ----》predicted as $predicted_output_index")
                label_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                target_index = [i for i in label_index if i != predicted_output_index]
                @assert (predicted_output_index - 1) == sample_label
                total_time = @elapsed begin
                    d = MIPVerify.robustness_checking_verification(
                        nn,
                        sample_image,
                        target_index,
                        Gurobi.Optimizer,
                        #OutputFlag=0, #prevents any output from being printed out
                        Dict("OutputFlag" => 0),
                        pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.15),
                        norm_order = Inf,
                        tightening_algorithm = lp,
                        radius = epsilon,
                    )
                end
                # solve_time = JuMP.solve_time(d[:Model])
                perturbed_sample_image = Float16.(value.(d[:PerturbedInput]))
                perturbation_predicted_output = perturbed_sample_image |> nn
                perturbation_predicted_index = argmax(perturbation_predicted_output)

                if (perturbation_predicted_index == predicted_output_index)
                    #发现虚假反例
                    false_adv_res = VerifyInfo(index, total_time, false, false, false, 0)
                    println("Find False adv\n")
                    push!(result, false_adv_res)
                else
                    #发现真实反例
                    true_adv_res = VerifyInfo(index, total_time, true, false, true, -1)
                    println("Find True adv\n")
                    push!(result, true_adv_res)
                end
            catch e
                #println("处理样本 $index 时出错: $e")
                println("verifed true\n")
                verified_robust = VerifyInfo(index, total_time, false, true, true, 1)
                push!(result, verified_robust)
            end
        end
        run_statistics = Dict(
            :result => result,
            :misclassified => misclassified,
            :round_error_class => round_error_class,
        )
        return run_statistics

    elseif data_type == 2
        misclassified = 0
        result = VerifyInfo[]
        round_error_class = Int[]
        mnist = MIPVerify.read_datasets(data_name)
        nn = MIPVerify.get_example_network_params(network_name)
        for index in min_data_num:max_data_num
            total_time = 0
            try
                ##参数读取部分代码
                sample_image = MIPVerify.get_image(mnist.test.images, index)
                sample_label = MIPVerify.get_label(mnist.test.labels, index)
                predicted_output = Float16.(sample_image) |> nn
                predicted_output_index = argmax(predicted_output)
                if predicted_output_index != sample_label + 1
                    println("Sample $index misclassified.")
                    misclassified = misclassified + 1
                    continue
                end
                println("--------verifying index:$index ---------")
                #println("the label : $sample_label ----》predicted as $predicted_output_index")
                label_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                target_index = [i for i in label_index if i != predicted_output_index]
                @assert (predicted_output_index - 1) == sample_label
                d = MIPVerify.robustness_checking_verification(
                    nn,
                    sample_image,
                    target_index,
                    Gurobi.Optimizer,
                    #OutputFlag=0, #prevents any output from being printed out
                    Dict("OutputFlag" => 0),
                    pp = MIPVerify.LInfNormBoundedPerturbationFamily(0.15),
                    norm_order = Inf,
                    tightening_algorithm = lp,
                    radius = epsilon,
                )
                # solve_time = JuMP.solve_time(d[:Model])
                perturbed_sample_image = Float16.(value.(d[:PerturbedInput]))
                perturbation_predicted_output = perturbed_sample_image |> nn
                perturbation_predicted_index = argmax(perturbation_predicted_output)

                if (perturbation_predicted_index == predicted_output_index)
                    #发现虚假反例
                    false_adv_res = VerifyInfo(index, total_time, false, false, false, 0)
                    println("Find False adv\n")
                    push!(result, false_adv_res)
                else
                    #发现真实反例
                    true_adv_res = VerifyInfo(index, total_time, true, false, true, -1)
                    println("Find True adv\n")
                    push!(result, true_adv_res)
                end

            catch e
                #println("处理样本 $index 时出错: $e")
                println("verifed true\n")
                verified_robust = VerifyInfo(index, total_time, false, true, true, 1)
                push!(result, verified_robust)
            end
        end
        run_statistics = Dict(
            :result => result,
            :misclassified => misclassified,
            :round_error_class => round_error_class,
        )
        return run_statistics
    end



end


return_dict = mp_robust_verify()
res=return_dict[:result]


data_name = ARGS[2]
network_name = ARGS[3]
min_data_num = parse(Int, ARGS[4])
max_data_num = parse(Int, ARGS[5])
epsilon = parse(Float64, ARGS[6])

# Find indexes where verified_res is not 1
verify_unknown = [i.index for i in res if i.verified==false && i.verified_res == 0]

# Find indexes where verified is true and verified_res is 1
verify_true = [i.index for i in res if i.verified==true && i.verified_res == 1]


verify_false = [i.index for i in res if i.verified==true && i.verified_res == -1]


total_num = max_data_num - min_data_num + 1 -return_dict[:misclassified]

# println("verified example: $verify_true")
# println("verified unknown example: $verify_unknown")

println("total: ",total_num)
println("uknown: ",length(verify_unknown))
println("true: ",length(verify_true))
println("false: ",length(verify_false))
#@assert(length(verify_unknown) + length(verify_true) == length(total_num))

# File path to save the results in the current directory
output_file = "./mp_verify_res/mpres_$(network_name)_$(data_name)_ep$(epsilon)_from_index$(min_data_num)to$(max_data_num).txt"

# Write the results to the file
open(output_file, "w") do file
    println(file, "verify_unknown: ", join(verify_unknown, ", "))
    println(file, "verify_true: ", join(verify_true, ", "))
    println(file, "verify_false: ", join(verify_false, ", "))
end