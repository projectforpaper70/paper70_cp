"""
$(SIGNATURES)

Makes named example neural networks available as a [`NeuralNet`](@ref) object.

# Arguments
* `name::String`: Name of example neural network. Options:
    * `'MNIST.n1'`:
        * Architecture: Two fully connected layers with 40 and 20 units.
        * Training: Trained regularly with no attempt to increase robustness.
    * `'MNIST.WK17a_linf0.1_authors'`.
        * Architecture: Two convolutional layers (stride length 2) with 16 and
          32 filters respectively (size 4 × 4 in both layers), followed by a
          fully-connected layer with 100 units.
        * Training: Network trained to be robust to attacks with \$l_\\infty\$ norm
          at most 0.1 via method in [Provable defenses against adversarial examples
          via the convex outer adversarial polytope](https://arxiv.org/abs/1711.00851).
          Is MNIST network for which results are reported in that paper.
    * `'MNIST.RSL18a_linf0.1_authors'`.
        * Architecture: One fully connected layer with 500 units.
        * Training: Network trained to be robust to attacks with \$l_\\infty\$ norm
          at most 0.1 via method in [Certified Defenses against Adversarial Examples
          ](https://arxiv.org/abs/1801.09344).
          Is MNIST network for which results are reported in that paper.
"""
function get_example_network_params(name::String)::NeuralNet
    if name == "MNIST.n1"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "n1.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 40))
        fc2 = get_matrix_params(param_dict, "fc2", (40, 20))
        logits = get_matrix_params(param_dict, "logits", (20, 10))

        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F32MNIST40"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "n1.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 40))
        fc2 = get_matrix_params(param_dict, "fc2", (40, 20))
        logits = get_matrix_params(param_dict, "logits", (20, 10))

        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F32MNIST24"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_dnn_fp32.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24))
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24))
        logits = get_matrix_params(param_dict, "logits", (24, 10))
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F16MNIST24"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F16Fashion24"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "fashion_mnist_model.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F16MNIST24to64"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16,to_double=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16,to_double=true)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16,to_double=true)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F16MNISTinput_88"
        param_dict = prep_data_file(joinpath("weights","mnist"),"resized_mnist88input_mnist_dnn_fp16.mat") |> matread 
        fc1 = get_matrix_params(param_dict, "fc1", (64, 10),data_type=Float16,to_double=true)
        fc2 = get_matrix_params(param_dict, "fc2", (10, 10),data_type=Float16,to_double=true)
        logits = get_matrix_params(param_dict, "logits", (10, 10),data_type=Float16,to_double=true)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F16MNISTinput_77"
        param_dict = prep_data_file(joinpath("weights","mnist"),"resized_mnist77input_mnist_dnn_fp16.mat") |> matread 
        fc1 = get_matrix_params(param_dict, "fc1", (49, 10),data_type=Float16)
        fc2 = get_matrix_params(param_dict, "fc2", (10, 10),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (10, 10),data_type=Float16)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "F32MNIST_INPUT77"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "resized77_mnist_dnn_fp32.mat") |> matread
        fc1 = get_matrix_params(param_dict,"fc1", (49,10))
        fc2 = get_matrix_params(param_dict, "fc2", (10, 10))
        logits = get_matrix_params(param_dict, "logits", (10, 10))
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), logits], name)
        return nn
    elseif name == "5layerF16MNIST24"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_5layers_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16)
        fc3 = get_matrix_params(param_dict, "fc3", (24, 24),data_type=Float16)
        fc4 = get_matrix_params(param_dict, "fc4", (24, 24),data_type=Float16)
        # fc7 = get_matrix_params(param_dict, "fc7", (24, 24),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16)
        #nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), fc3, ReLU(mip), fc4, ReLU(mip), fc5, ReLU(mip), fc6, ReLU(mip), fc7, ReLU(mip), logits], name)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), fc3, ReLU(mip),fc4, ReLU(mip),logits], name)
        return nn
    elseif name =="8layerF16MNIST24"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_8layers_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16)
        fc3 = get_matrix_params(param_dict, "fc3", (24, 24),data_type=Float16)
        fc4 = get_matrix_params(param_dict, "fc4", (24, 24),data_type=Float16)
        fc5 = get_matrix_params(param_dict, "fc5", (24, 24),data_type=Float16)
        fc6 = get_matrix_params(param_dict, "fc6", (24, 24),data_type=Float16)
        fc7 = get_matrix_params(param_dict, "fc7", (24, 24),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16)
        nn = Sequential([
            Flatten(4),
            fc1,
            ReLU(MIPVerify.interval_arithmetic),
            fc2,
            ReLU(MIPVerify.mip),
            fc3,
            ReLU(MIPVerify.mip),
            fc4,
            ReLU(MIPVerify.mip),
            fc5,
            ReLU(MIPVerify.mip),
            fc6,
            ReLU(MIPVerify.mip),
            fc7,
            ReLU(MIPVerify.mip),
            logits,
            ], "8layerF16MNIST24")
        #: ",typeof(nn_itv))
        return nn
    elseif name =="8layerF16MNIST24_train64"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_8layers_dnn_fp64.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16)
        fc3 = get_matrix_params(param_dict, "fc3", (24, 24),data_type=Float16)
        fc4 = get_matrix_params(param_dict, "fc4", (24, 24),data_type=Float16)
        fc5 = get_matrix_params(param_dict, "fc5", (24, 24),data_type=Float16)
        fc6 = get_matrix_params(param_dict, "fc6", (24, 24),data_type=Float16)
        fc7 = get_matrix_params(param_dict, "fc7", (24, 24),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16)
        nn = Sequential([
            Flatten(4),
            fc1,
            ReLU(MIPVerify.interval_arithmetic),
            fc2,
            ReLU(MIPVerify.mip),
            fc3,
            ReLU(MIPVerify.mip),
            fc4,
            ReLU(MIPVerify.mip),
            fc5,
            ReLU(MIPVerify.mip),
            fc6,
            ReLU(MIPVerify.mip),
            fc7,
            ReLU(MIPVerify.mip),
            logits,
            ], "8layerF16MNIST24_train64")
        #: ",typeof(nn_itv))
        return nn
    elseif name == "5layerF16MNIST24to64"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_5layers_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16,to_double=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16,to_double=true)
        fc3 = get_matrix_params(param_dict, "fc3", (24, 24),data_type=Float16,to_double = true)
        fc4 = get_matrix_params(param_dict, "fc4", (24, 24),data_type=Float16,to_double = true)
        # fc7 = get_matrix_params(param_dict, "fc7", (24, 24),data_type=Float16)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16,to_double = true)
        #nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), fc3, ReLU(mip), fc4, ReLU(mip), fc5, ReLU(mip), fc6, ReLU(mip), fc7, ReLU(mip), logits], name)
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(mip), fc3, ReLU(mip),fc4, ReLU(mip),logits], name)
        return nn
    elseif name == "MNIST50"
        nnet_param=NNetFormat(prep_data_file(joinpath("nnet_model","mnist"),"mnist50.nnet"),"MNIST")
        fc1 = get_matrix_params_nnet(nnet_param,1,(784,50))
        fc2 = get_matrix_params_nnet(nnet_param,2,(50,50))
        logits = get_matrix_params_nnet(nnet_param,3,(50,10))
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(),logits],"MNIST50")
        return nn
    elseif name == "MNIST512"
        nnet_param=NNetFormat(prep_data_file(joinpath("nnet_model","mnist"),"mnist512.nnet"),"MNIST")
        fc1 = get_matrix_params_nnet(nnet_16param,2,(512,512))
        logits = get_matrix_params_nnet(nnet_param,3,(512,10))
        nn = Sequential([Flatten(4), fc1, ReLU(interval_arithmetic), fc2, ReLU(),logits],"MNIST512")
        return nn
    elseif name == "MNIST.WK17a_linf0.1_authors"
        param_dict =
            prep_data_file(
                joinpath("weights", "mnist", "WK17a", "linf0.1"),
                "master_seed_1_epochs_100.mat",
            ) |> matread
        conv1 = get_conv_params(param_dict, "conv1", (4, 4, 1, 16), expected_stride = 2)
        conv2 = get_conv_params(param_dict, "conv2", (4, 4, 16, 32), expected_stride = 2)
        fc1 = get_matrix_params(param_dict, "fc1", (1568, 100))
        logits = get_matrix_params(param_dict, "logits", (100, 10))

        nn = Sequential(
            [
                conv1,
                ReLU(interval_arithmetic),
                conv2,
                ReLU(),
                Flatten([1, 3, 2, 4]),
                fc1,
                ReLU(),
                logits,
            ],
            name,
        )
        return nn
    elseif name == "MNIST.RSL18a_linf0.1_authors"
        param_dict =
            prep_data_file(joinpath("weights", "mnist", "RSL18a", "linf0.1"), "two-layer.mat") |>
            matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 500))
        logits = get_matrix_params(param_dict, "logits", (500, 10))

        nn = Sequential([Flatten([1, 3, 2, 4]), fc1, ReLU(interval_arithmetic), logits], name)
        return nn
    else
        throw(DomainError("No example network named $name."))
    end
end




function get_example_network_params_withSageSolver(name::String,model_pyobj::PyObject)::NeuralNet
    if name =="F16MNIST24itv"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16,trans2itv=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16,trans2itv=true)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16,trans2itv=true)
        nn_itv = SequentialItv([
            fc1,
            ReLU(MIPVerify.interval_arithmetic,model_pyobj),
            fc2,
            ReLU(MIPVerify.mip,model_pyobj),
            logits,
            ], "F16MNIST24itv")
        #println("type is: ",typeof(nn_itv))
        return nn_itv
    elseif name =="F32MNIST24itv"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_dnn_fp32.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float32,trans2itv=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float32,trans2itv=true)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float32,trans2itv=true)
        nn_itv = SequentialItv([
            fc1,
            ReLU(MIPVerify.interval_arithmetic,model_pyobj),
            fc2,
            ReLU(MIPVerify.mip,model_pyobj),
            logits,
            ], "F32MNIST24itv")
        #println("type is: ",typeof(nn_itv))
        return nn_itv
    elseif name =="8layerF16MNIST24itv"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_8layers_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16,trans2itv=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16,trans2itv=true)
        fc3 = get_matrix_params(param_dict, "fc3", (24, 24),data_type=Float16,trans2itv=true)
        fc4 = get_matrix_params(param_dict, "fc4", (24, 24),data_type=Float16,trans2itv=true)
        fc5 = get_matrix_params(param_dict, "fc5", (24, 24),data_type=Float16,trans2itv=true)
        fc6 = get_matrix_params(param_dict, "fc6", (24, 24),data_type=Float16,trans2itv=true)
        fc7 = get_matrix_params(param_dict, "fc7", (24, 24),data_type=Float16,trans2itv=true)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16,trans2itv=true)
        nn_itv = SequentialItv([
            fc1,
            ReLU(MIPVerify.interval_arithmetic,model_pyobj),
            fc2,
            ReLU(MIPVerify.mip,model_pyobj),
            fc3,
            ReLU(MIPVerify.mip,model_pyobj),
            fc4,
            ReLU(MIPVerify.mip,model_pyobj),
            fc5,
            ReLU(MIPVerify.mip,model_pyobj),
            fc6,
            ReLU(MIPVerify.mip,model_pyobj),
            fc7,
            ReLU(MIPVerify.mip,model_pyobj),
            logits,
            ], "F16MNIST24itv")
        #: ",typeof(nn_itv))
        return nn_itv
    elseif name =="5layerF16MNIST24itv"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "mnist_8layers_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16,trans2itv=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16,trans2itv=true)
        fc3 = get_matrix_params(param_dict, "fc3", (24, 24),data_type=Float16,trans2itv=true)
        fc4 = get_matrix_params(param_dict, "fc4", (24, 24),data_type=Float16,trans2itv=true)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16,trans2itv=true)
        nn_itv = SequentialItv([
            fc1,
            ReLU(MIPVerify.interval_arithmetic,model_pyobj),
            fc2,
            ReLU(MIPVerify.mip,model_pyobj),
            fc3,
            ReLU(MIPVerify.mip,model_pyobj),
            fc4,
            ReLU(MIPVerify.mip,model_pyobj),
            logits,
            ], "F16MNIST24itv")
        return nn_itv
    elseif name == "F16MNISTinput_77itv"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "resized_mnist77input_mnist_dnn_fp16.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (49, 10),data_type=Float16,trans2itv=true)
        fc2 = get_matrix_params(param_dict, "fc2", (10, 10),data_type=Float16,trans2itv=true)
        logits = get_matrix_params(param_dict, "logits", (10, 10),data_type=Float16,trans2itv=true)
        nn_itv = SequentialItv([
            fc1,
            ReLU(MIPVerify.interval_arithmetic,model_pyobj),
            fc2,
            ReLU(MIPVerify.mip,model_pyobj),
            logits,
            ], "F16MNISTinput_77itv")
        #println("type is: ",typeof(nn_itv))
        return nn_itv
    elseif name =="F16Fashion784itv"
        param_dict = prep_data_file(joinpath("weights", "mnist"), "fashion_mnist_model.mat") |> matread
        fc1 = get_matrix_params(param_dict, "fc1", (784, 24),data_type=Float16,trans2itv=true)
        fc2 = get_matrix_params(param_dict, "fc2", (24, 24),data_type=Float16,trans2itv=true)
        logits = get_matrix_params(param_dict, "logits", (24, 10),data_type=Float16,trans2itv=true)
        nn_itv = SequentialItv([
            fc1,
            ReLU(MIPVerify.interval_arithmetic,model_pyobj),
            fc2,
            ReLU(MIPVerify.mip,model_pyobj),
            logits,
            ], "F16Fashion784itv")
        #println("type is: ",typeof(nn_itv))
        return nn_itv
end
    
end

# TODO (vtjeng): Add mnist networks Ragunathan/Steinhardt/Liang.
# TODO (vtjeng): Make network naming case insensitive.
