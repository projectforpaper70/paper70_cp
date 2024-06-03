using FileIO
using Images

export load_all_binary_images, load_specific_binary_image

const data_repo_path = "https://github.com/vtjeng/MIPVerify_data/raw/master"

function prep_data_file(relative_dir::String, filename::String)::String
    absolute_dir = joinpath(dependencies_path, relative_dir)
    if !ispath(absolute_dir)
        mkpath(absolute_dir)
    end

    relative_file_path = joinpath(relative_dir, filename)
    absolute_file_path = joinpath(dependencies_path, relative_file_path)
    if !isfile(absolute_file_path)
        if Sys.iswindows()
            # On windows, the `joinpath` command uses `\` as a separator.
            # TODO: This is a bit of a hack; we might prefer rethinking `relative_dir`
            relative_url_path = replace(relative_file_path, "\\" => "/")
        else
            relative_url_path = relative_file_path
        end
        url = string(data_repo_path, "/", relative_url_path)
        download(url, absolute_file_path)
    end

    return absolute_file_path
end




function load_all_binary_images(binary_file_path::AbstractString, image_size::Tuple{Int, Int}, num_images::Int)
    # Open the binary file
    file = open(binary_file_path, "r")

    # Initialize arrays to store images and labels
    images = Vector{Matrix{Float32}}(undef, num_images)
    labels = Vector{Int}(undef, num_images)

    # Iterate over each image
    for i in 1:num_images
        # Read image size
        image_size_bytes = read(file, 8)
        width, height = reinterpret(Int32, image_size_bytes)

        # Read label
        label_bytes = read(file, 4)
        label = reinterpret(Int32, label_bytes)[1]  # Get the first Float32 value

        # Read image data
        image_bytes = read(file, 4 * image_size[1] * image_size[2])

        # Reshape image data to image_size matrix
        image_data = Matrix{Float32}(undef, image_size[1], image_size[2])
        for j in 1:image_size[1]
            for k in 1:image_size[2]
                pixel_start = (j - 1) * image_size[1] * 4 + (k - 1) * 4 + 1
                image_pixel = reinterpret(Float32, image_bytes[pixel_start:pixel_start+3])[1]  # Get the first Float32 value
                image_data[j, k] = image_pixel
            end
        end

        # Store image and label
        images[i] = image_data
        labels[i] = label
    end

    # Close the binary file
    close(file)

    return images, labels
end



function load_specific_binary_image(binary_file_path::AbstractString, image_size::Tuple{Int, Int}, index::Int)
    # Open the binary file
    file = open(binary_file_path, "r")

    # Initialize arrays to store images and labels
    images = Matrix{Float32}(undef, image_size[1], image_size[2])
    label = Int(0)

    # Seek to the start position of the desired image
    seek(file, (index - 1) * (8 + 4 + 4 * image_size[1] * image_size[2]))

    # Read image size
    image_size_bytes = read(file, 8)
    width, height = reinterpret(Int32, image_size_bytes)

    # Read label
    label_bytes = read(file, 4)
    label = reinterpret(Int32, label_bytes)[1]  # Get the first Float32 value

    # Read image data
    image_bytes = read(file, 4 * image_size[1] * image_size[2])

    # Reshape image data to image_size matrix
    for j in 1:image_size[1]
        for k in 1:image_size[2]
            pixel_start = (j - 1) * image_size[1] * 4 + (k - 1) * 4 + 1
            image_pixel = reinterpret(Float32, image_bytes[pixel_start:pixel_start+3])[1]  # Get the first Float32 value
            images[j, k] = image_pixel
        end
    end

    # Close the binary file
    close(file)

    return images, label
end


# #usage_example
# # Example usage:
# binary_file_path = "resized_images/resized_mnist_images77_test.bin"
# image_size = (7, 7)
# num_images = 10000
# images, labels = load_all_binary_images(binary_file_path, image_size, num_images)

# # Display the first image and its label
# @printf("Label: %d\n", labels[102])  # Changed %.0f to %d for integer label
# Gray.(images[102]*10)  # Display the image




# # Define the path to the binary file
# binary_file_path = "resized_images/resized_mnist_images88_train.bin"

# # Define the image size
# image_size = (8, 8)

# # Specify the index of the image you want to read
# index = 1000  # Example index

# # Load the specified image and its label
# image, label = load_specific_binary_image(binary_file_path, image_size, index)

# # Display the image and its label
# println("Label: ", label)
# Gray.(image)  # Display the image
