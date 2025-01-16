import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
import albumentations as album
import config


def leaky_relu(x):
    return np.where(x > 0, x, .01 * x)


def derivative_leaky_relu(x):
    return np.where(x > 0, 1, .01)


def leaky_relu_backward(derivative_output, input):
    return derivative_output * derivative_relu(input)


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return x > 0


def relu_backward(derivative_output, input):
    return derivative_output * derivative_relu(input)



def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def softmax_derivative(output, labels):
    # output must be result of softmax function and have same shape as labels
    # Gradient of softmax with cross-entropy loss
    grad = output - labels

    return grad


def derivative_tanh(x):
    return 1 - np.tanh(x) ** 2


def tanh_backward(derivative_output, input):
    return derivative_output * derivative_tanh(input)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    calculation = sigmoid(x)
    return calculation * (1 - calculation)


def sigmoid_backward(derivative_output, input):
    return derivative_output * derivative_sigmoid(input)


def initialize_predefined_kernels(conv_layers):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])



    filters = [sobel_x, sobel_y, scharr_x, scharr_y, laplacian]

    # Stack/repeat num_input_channels channels
    if conv_layers == 1:
        kernels = [np.stack([np.repeat(f[:, :, np.newaxis], 3, axis=2) for f in filters], axis=-1), None]
    elif conv_layers == 2:
        kernels = [np.stack([np.repeat(f[:, :, np.newaxis], 3, axis=2) for f in filters], axis=-1), np.stack([np.repeat(f[:, :, np.newaxis], 3, axis=2) for f in filters], axis=-1)]
    else:
        print("Invalid number of convolution/pooling layers. Please choose 1 or 2.")
        exit(0)

    # Get the number of kernels
    num_kernels = kernels[0].shape[-1]

    return kernels, num_kernels


###############################################################################################################
def place_holder_kernels():
    # Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    sobel_135 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

    # Prewitt filters
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_45 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])
    prewitt_135 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])

    # Scharr filters
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

    # Laplacian filter
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Roberts Cross filters
    roberts_cross_x = np.array([[1, 0], [0, -1]])
    roberts_cross_y = np.array([[0, 1], [-1, 0]])

    # Kirsch Compass filters
    kirsch_north = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    kirsch_east = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    kirsch_south = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    kirsch_west = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])

    # List of filters
    filters = [sobel_x, sobel_y, sobel_45, sobel_135,
               prewitt_x, prewitt_y, prewitt_45, prewitt_135,
               scharr_x, scharr_y, laplacian,
               roberts_cross_x, roberts_cross_y,
               kirsch_north, kirsch_east, kirsch_south, kirsch_west]

    blur_kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]) / 9.0

    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    horizontal_edge = np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]])

    vertical_edge = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]])

    identity = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])

    # Repeat across 3 channels (RGB) and stack them
    kernels = np.stack([np.repeat(f[:, :, np.newaxis], 3, axis=2) for f in filters], axis=-1)

    # Get the number of kernels
    num_kernels = kernels.shape[-1]

    return kernels, num_kernels
####################################################################################################################


def convolution(image, kernel, stride=1, padding=0):

    if len(image.shape) == 4:
        # Padding should only be applied to the height and width dimensions
        if padding > 0:
            image = np.pad(image, [(padding, padding), (padding, padding), (0, 0), (0, 0)], mode='constant')
    else:
        # Normal case with 3D image (height, width, channels)
        if padding > 0:
            image = np.pad(image, [(padding, padding), (padding, padding), (0, 0)], mode='constant')

    # Dimensions of output
    output_dim = (image.shape[0] - kernel.shape[0]) // stride + 1
    output = np.zeros((output_dim, output_dim, image.shape[2], kernel.shape[-1]))

    # Perform convolution
    for h in range(output_dim):
        for w in range(output_dim):
            region = image[h * stride:h * stride + kernel.shape[0], w * stride:w * stride + kernel.shape[1], :]
            # all kernels at once
            output[h, w] = np.sum(region[:, :, :, np.newaxis] * kernel, axis=(0, 1, 2))

    return output



def max_pooling(image, size=2, stride=2):
    # Output dimension
    output_dim = (image.shape[0] - size) // stride + 1
    output = np.zeros((output_dim, output_dim, image.shape[2], image.shape[-1]))  # Output initialized to zeros

    # Perform operation
    for h in range(output_dim):
        for w in range(output_dim):
            region = image[h * stride:h * stride + size, w * stride:w * stride + size, :, :]
            output[h, w, :, :] = np.max(region, axis=(0, 1))

    return output


'''
Take 1D array and transform into 2D array
Example:
    1D Array: (5, 4, 1, 3, 0, 4, 3)
    2D one_hot_encoded:
              0  1  2  3  4  5
              ----------------
    img 1    [0, 0, 0, 0, 0, 1]
    img 2    [0, 0, 0, 0, 1, 0]
    img 3    [0, 1, 0, 0, 0, 0]
    img 4    [0, 0, 0, 1, 0, 0]
    img 5    [1, 0, 0, 0, 0, 0]
    img 6    [0, 0, 0, 0, 1, 0]
    img 7    [0, 0, 0, 1, 0, 0]

'''
def one_hot_encode(data, num_classes):
    transformed_labels = np.zeros((data.size, num_classes))
    transformed_labels[np.arange(data.size), data.flatten()] = 1
    return transformed_labels


"""
Display image and label
"""
def display_image(image, label, number):
    plt.imshow(image[number])
    plt.title(f"Label: {label[number]}")
    plt.show()


def remap_classes(labels, neither_label, classes_to_train):
    if neither_label == "Y":
        labels_mapped = np.full_like(labels, 99)

        # Map classes to respective indices
        # Example: classes_to_train = [2, 3, 4, 5, 6, 7]
        #   neither ->  99
        #       2   ->  1
        #       3   ->  2
        #       4   ->  3
        #       5   ->  4
        #       6   ->  5
        #       7   ->  6
        for i, class_val in enumerate(classes_to_train, start=1):
            labels_mapped[labels == class_val] = i

        # Remap neither_label(-99) to 0
        labels_mapped[labels_mapped == neither_label] = 0

        return labels_mapped
    else:
        labels_mapped = np.full_like(labels, 0)

        # Map classes to respective indices
        # Example: classes_to_train = [2, 3, 4, 5, 6, 7]
        #       2   ->  1
        #       3   ->  2
        #       4   ->  3
        #       5   ->  4
        #       6   ->  5
        #       7   ->  6
        for i, class_val in enumerate(classes_to_train, start=0):
            labels_mapped[labels == class_val] = i

        return labels_mapped


def normalize_images(images):
    return images / 255.0


def fully_connected_operation(x, weights, bias):
    return np.dot(x, weights) + bias


'''
def initialize_weights(num_output=6, num_neurons=20, num_pooling=1, weight_scale=0.01):
    # Generate pre-defined kernels
    # Shape (3, 3, 3, 5) 3 x 3 matrix, 3 color channels, 5 kernels
    conv_kernels, num_kernels = initialize_predefined_kernels(num_pooling)

    # Flatten output size
    # 32 from cifar-10 32x32 size
    # 3 from rgb
    flatten_size = int((32 / (num_pooling * 2)) ** 2 * num_kernels * 3)

    # Adjust fully connected layer weight dimensions based on flattened output size (1792), neurons (128), output neurons (10
    fully_connected_weights = np.random.randn(flatten_size, num_neurons) * np.sqrt(2 / flatten_size)
    fully_connected_bias = np.random.randn(num_neurons) * weight_scale
    output_weights = np.random.randn(num_neurons, num_output) * np.sqrt(2 / num_neurons)
    output_bias = np.random.randn(num_output) * weight_scale

    return conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias
'''
def initialize_weights(num_output=7, num_neurons=20, num_pooling=1, weight_scale=0.01):
    # Generate pre-defined kernels
    # Shape (3, 3, 3, 7) 3 x 3 matrix, 3 color channels, 7 kernels
    conv_kernels, num_kernels = initialize_predefined_kernels(num_pooling)

    # Flatten output size
    # 32 from cifar-10 32x32 size
    # 3 from rgb
    flatten_size = int((32 / (num_pooling * 2)) ** 2 * num_kernels * 3)

    # Adjust fully connected layer weight dimensions based on flattened output size (1792), neurons (128), output neurons (10
    fully_connected_weights = np.random.randn(flatten_size, num_neurons) * weight_scale
    fully_connected_bias = np.random.randn(num_neurons) * weight_scale
    output_weights = np.random.randn(num_neurons, num_output) * weight_scale
    output_bias = np.random.randn(num_output) * weight_scale

    return conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias


def set_activation_function(activation_name):
    # Choose activation functions
    if activation_name == 'relu':
        activation_fn = relu
        activation_backward_fn = relu_backward
    elif activation_name == 'leaky_relu':
        activation_fn = leaky_relu
        activation_backward_fn = leaky_relu_backward
    elif activation_name == 'tanh':
        activation_fn = tanh
        activation_backward_fn = tanh_backward
    elif activation_name == 'sigmoid':
        activation_fn = sigmoid
        activation_backward_fn = sigmoid_backward
    else:
        raise ValueError("Invalid activation function")

    return activation_fn, activation_backward_fn


def cross_entropy_loss(predictions, labels):
    # predictions in range between (epsilon, 1-epsilon)
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

    loss = -np.log(predictions[np.argmax(labels)])

    return loss


def update_parameters(params, grads, learning_rate):
    for i in range(len(params)):
        params[i] -= learning_rate * grads[i]


def fully_connected_backward(output_gradient, fully_connected_input, fully_connected_weights):
    # Ensure the input and gradient have the correct dimensions (2D) (1, num features)
    fully_connected_input = fully_connected_input.reshape(1, -1)
    output_gradient = output_gradient.reshape(1, -1)

    # gradients for weights, biases, inputs
    gradient_weights = np.dot(fully_connected_input.T, output_gradient)
    gradient_biases = np.sum(output_gradient, axis=0)
    gradient_input = np.dot(output_gradient, fully_connected_weights.T)

    return gradient_input, gradient_weights, gradient_biases


def backward_pass(output, label, fully_connected_weights, output_weights, activation_function_backward,
                  forward_pass_parameters):
    # Loss calculation
    loss_value = cross_entropy_loss(output, label)

    # softmax derivative
    softmax_gradient = softmax_derivative(output, label)

    # fully connected backprop
    grad_activ_out_layer, grad_out_weights, grad_out_biases = fully_connected_backward(softmax_gradient,
                                                                                       forward_pass_parameters[0],
                                                                                       output_weights)
    grad_activ_func = activation_function_backward(grad_activ_out_layer, forward_pass_parameters[1])
    grad_activ_fully_connected_layer, grad_fully_connected_weights, grad_fully_connected_biases = fully_connected_backward(
        grad_activ_func, forward_pass_parameters[2], fully_connected_weights)

    # Return gradients for the fully connected layers
    return [grad_fully_connected_weights, grad_fully_connected_biases, grad_out_weights, grad_out_biases], loss_value


def forward_pass(image, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, conv_activation_fn, fc_activation_fn, num_conv_pool_layers):
    # 1st conv layers and max pooling
    convolution_output = convolution(image, conv_kernels[0], stride=1, padding=1)
    activation_output = conv_activation_fn(convolution_output)
    pool_output = max_pooling(activation_output, size=2, stride=2)


    # 2nd conv layers and max pooling
    if num_conv_pool_layers == 2:
        convolution_output = convolution(pool_output, conv_kernels[1], stride=1, padding=1)
        activation_output = conv_activation_fn(convolution_output)
        pool_output = max_pooling(activation_output, size=2, stride=2)

    flattened_output = pool_output.flatten()

    # Pass through fully connected layers
    fully_connected_output = fully_connected_operation(flattened_output, fully_connected_weights, fully_connected_bias)
    activation_output = fc_activation_fn(fully_connected_output)

    # Final output result
    output = softmax(fully_connected_operation(activation_output, output_weights, output_bias))

    return output, [activation_output, fully_connected_output, flattened_output]


def train_network(train_images, train_labels, num_output, num_neurons=128, num_epochs=5, learning_rate=0.01,
                  weight_scale=0.01, conv_activation_func='leaky_relu', fc_activation_func='relu', num_conv_pool_layers = 1):
    # Augmentation settings
    transform = album.Compose([
        album.HorizontalFlip(p=0.5),
        album.Rotate(limit=10, p=0.5),
        album.RandomScale(scale_limit=0.1, p=0.5),
        album.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        album.Resize(32, 32)
    ])

    # Initialize weights
    conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias = initialize_weights(
        num_output=num_output,
        num_neurons=num_neurons,
        num_pooling=num_conv_pool_layers,
        weight_scale=weight_scale)

    # Initialize activation functions
    conv_activation_fn, conv_activation_backward_fn = set_activation_function(conv_activation_func)
    fc_activation_fn, fc_activation_backward_fn = set_activation_function(fc_activation_func)

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0

        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        # Shuffle train_images + train_labels
        indices = np.random.permutation(len(train_images))
        train_images = train_images[indices]
        train_labels = train_labels[indices]

        for i in range(len(train_images)):
            image = train_images[i]
            label = train_labels[i]

            # apply augmentation to image
            augmented = transform(image=image)
            augmented_image = augmented['image']

            # forward pass
            output, forward_pass_params = forward_pass(augmented_image, conv_kernels, fully_connected_weights, fully_connected_bias,
                                                       output_weights, output_bias, conv_activation_fn, fc_activation_fn, num_conv_pool_layers)

            # backward pass, returning gradients for fully connected layer
            # returns [deriv_fully_connected_weights, deriv_fully_connected_bias, deriv_output_weights, deriv_output_bias], loss
            gradients, loss = backward_pass(output, label, fully_connected_weights, output_weights,
                                            fc_activation_backward_fn, forward_pass_params)

            # Add loss to total loss
            total_loss += loss

            # Update fully connected layers weights
            params = [fully_connected_weights, fully_connected_bias, output_weights, output_bias]
            update_parameters(params, gradients, learning_rate)

            # Compute accuracy
            if np.argmax(output) == np.argmax(label):
                correct_predictions += 1

        accuracy = correct_predictions / len(train_images)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {total_loss / len(train_images)}, Accuracy: {accuracy}")

        # Test purposes
        #if (epoch+1) % test_every_x_epoch == 0:
        #    test_network(test_images, test_labels, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, activation_fn, num_conv_pool_layers, folderpath, (epoch+1))

    return conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, conv_activation_fn, fc_activation_fn


def test_network(test_images, test_labels, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, conv_activation_fn, fc_activation_fn, num_conv_pool_layers, folderpath, test_num):
    if folderpath is not None:
        csv_filepath = os.path.join(folderpath, f"results_{test_num}.csv")
        file = open(csv_filepath, mode="w", newline="")
        writer = csv.writer(file)
        writer.writerow(["Output", "Label", "Output Argmax", "Label Argmax", "Accuracy"])

        # Test trained model
        correct_predictions = 0
        for i in range(len(test_images)):
            image = test_images[i]
            label = test_labels[i]
            output, _ = forward_pass(image, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, conv_activation_fn, fc_activation_fn, num_conv_pool_layers)
            writer.writerow([output, label, np.argmax(output), np.argmax(label), None])
            if np.argmax(output) == np.argmax(label):
                correct_predictions += 1
        test_accuracy = correct_predictions / len(test_images)
        writer.writerow([None, None, None, None, test_accuracy])
        file.close()
        print(f"Test Accuracy: {test_accuracy}")
    else:
        # Test trained model
        correct_predictions = 0
        for i in range(len(test_images)):
            image = test_images[i]
            label = test_labels[i]
            output, _ = forward_pass(image, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights,
                                     output_bias, conv_activation_fn, fc_activation_fn, num_conv_pool_layers)
            if np.argmax(output) == np.argmax(label):
                correct_predictions += 1
        test_accuracy = correct_predictions / len(test_images)
        print(f"Test Accuracy: {test_accuracy}")


def create_folder(test_name):
    if test_name is None:
        return None
    folder_path = os.path.join(os.getcwd(), test_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        print("Test name already exists.\nExiting...")
        exit(0)
    return folder_path


def save_model(conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, folderpath):
    model = {
        "conv_kernels": conv_kernels,
        "fully_connected_weights": fully_connected_weights,
        "fully_connected_bias": fully_connected_bias,
        "output_weights": output_weights,
        "output_bias": output_bias
    }
    filepath = os.path.join(folderpath, f"model.pkl")
    with open(filepath, "wb") as file:
        pickle.dump(model, file)


def load_model(folder_path):
    filepath = os.path.join(folder_path, f"model.pkl")
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model["conv_kernels"], model["fully_connected_weights"], model["fully_connected_bias"], model["output_weights"], model["output_bias"]


def save_preset_values(folderpath, num_conv_pool_layers, conv_activation_string, fc_activation_func, num_neurons, num_epochs, learning_rate, weight_scale, neither_label):
    preset_values_filepath = os.path.join(folderpath, "preset_values.txt")
    with open(preset_values_filepath, "w") as file:
        file.write(f"Number of convolution/pooling layers: {num_conv_pool_layers}\n")
        file.write(f"Convolution Activation function: {conv_activation_string}\n")
        file.write(f"Fully Connected Activation function: {fc_activation_func}\n")
        file.write(f"Number of neurons: {num_neurons}\n")
        file.write(f"Number of epochs: {num_epochs}\n")
        file.write(f"Learning rate: {learning_rate}\n")
        file.write(f"Weight scale: {weight_scale}\n")
        file.write(f"Neither label: {'Yes' if neither_label == 'Y' else 'No'}\n")


def get_preset_values(filepath):
    preset_values_filepath = os.path.join(filepath, "preset_values.txt")
    with open(preset_values_filepath, "r") as file:
        lines = file.readlines()
        first_line = lines[0]
        second_line = lines[1]
        third_line = lines[2]
        num_conv_pool_layers = first_line.split(":")[1].strip()
        conv_activation_string = second_line.split(":")[1].strip()
        fc_activation_string = third_line.split(":")[1].strip()
        return conv_activation_string, fc_activation_string, int(num_conv_pool_layers)


def split_into_batches(images, labels, batch_size):
    image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    label_batches = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    return image_batches, label_batches


def main(activ_str, learning_rate, learn_rate_str, test_num):
    # Load CIFAR-10 dataset
    # train_images, train_labels : 50,000
    # test_images, test_labels : 10,000
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Shut keras up
    from keras.src.datasets import cifar10

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    ##############################################################################################################
    # Load a network
    load_file = config.load_file         # 'Y' or 'N'
    path = config.path

    # NETWORK VARIABLES
    num_conv_pool_layers = config.num_conv_pool_layers
    conv_activation = config.conv_activation
    num_neurons = config.num_neurons
    num_epochs = config.num_epochs
    weight_scale = config.weight_scale

    # test_name = "test1"
    test_name = config.test_name

    fc_activation = activ_str     # tanh, sigmoid, relu, leaky_relu
    learning_rate = learning_rate

    # 0 : Airplane
    # 1 : automobile
    # 2 : bird
    # 3 : cat
    # 4:  deer
    # 5 : dog
    # 6 : frog
    # 7 : horse
    # 8 : ship
    # 9 : truck
    # Classes to test: Animals
    classes_to_train = [2, 3, 4, 5, 6, 7]

    # To include all classes in classification ('Y') (non chosen classes will be relabeled to 0 as the neither class and be included as an output neruon) or only chosen classes
    neither_label = 'N'  # 'Y' OR 'N'

    # Speed up testing
    if neither_label == 'Y':
        train_num = 50000    # Max: 50000
        test_num = 10000     # Max: 10000
        train_images = train_images[:train_num]
        train_labels = train_labels[:train_num]
        test_images = test_images[:test_num]
        test_labels = test_labels[:test_num]
    elif neither_label == 'N':
        train_mask = np.isin(train_labels, classes_to_train)
        train_images = train_images[train_mask.flatten()]
        train_labels = train_labels[train_mask.flatten()]

        test_mask = np.isin(test_labels, classes_to_train)
        test_images = test_images[test_mask.flatten()]
        test_labels = test_labels[test_mask.flatten()]

        train_num = config.train_num   # Max: 30000
        test_num = config.test_num    # Max: 6000
        train_images = train_images[:train_num]
        train_labels = train_labels[:train_num]
        test_images = test_images[:test_num]
        test_labels = test_labels[:test_num]
    ##############################################################################################################

    if classes_to_train:
        print("Remapping labels...")
        train_labels_remapped = remap_classes(train_labels, neither_label, classes_to_train)
        test_labels_remapped = remap_classes(test_labels, neither_label, classes_to_train)
        print("Remapping labels complete.")


        print("Normalizing images...")
        train_images = normalize_images(train_images)
        test_images = normalize_images(test_images)
        print("Normalization complete.")

        print("One-hot encoding labels...")
        if neither_label == 'Y':
            num_classes = len(classes_to_train) + 1  # +1 bc neither_label
        else:
            num_classes = len(classes_to_train)
        train_labels = one_hot_encode(train_labels_remapped, num_classes)
        test_labels = one_hot_encode(test_labels_remapped, num_classes)
        print("One-hot encoding complete.")

        print("Separating test data into batches.")
        test_batch_images, test_batch_labels = split_into_batches(test_images, test_labels, 600)
        print("Separation complete.")
    else:
        raise ValueError("Please provide classes to train.")


    # Display images with corresponding label and indice
    # 0 : bird
    # 1 : cat
    # 2: deer
    # 3 : dog
    # 4 : frog
    # 5 : horse
    # [0, 1, 2, 3, 4, 5]
    # x = 2
    # i = 599
    # print(f"Index: {np.argmax(train_labels[i])}")
    # display_image(train_images, train_labels, i)
    # print(f"Index: {np.argmax(test_labels[x][i])}")
    # display_image(test_images[x], test_labels[x], i)

    if load_file == 'N':
        # Train a network
        conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, conv_activation_fc, fc_activation_func = train_network(
            train_images, train_labels, num_classes,
            num_neurons=num_neurons,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_scale=weight_scale,
            conv_activation_func=conv_activation,
            fc_activation_func=fc_activation,
            num_conv_pool_layers=num_conv_pool_layers)

        folder = create_folder(test_name)
        save_preset_values(folder, num_conv_pool_layers, conv_activation, fc_activation, num_neurons, num_epochs, learning_rate,
                           weight_scale, neither_label)
        save_model(conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, folder)

        """test_network(test_images, test_labels, conv_kernels, fully_connected_weights, fully_connected_bias,
                     output_weights, output_bias, conv_activation_fc, fc_activation_func, num_conv_pool_layers, folder,
                     1)"""

        # Test Network
        print("Testing data on trained model...")
        for i, (image_batch, label_batch) in enumerate(zip(test_batch_images, test_batch_labels)):
            print(f"Processing batch {i + 1}...")
            test_network(image_batch, label_batch, conv_kernels, fully_connected_weights, fully_connected_bias,
                         output_weights, output_bias, conv_activation_fc, fc_activation_func, num_conv_pool_layers, folder, i+1)


        # Test a trained network
        # test_network(test_images, test_labels, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, activation_func, num_conv_pool_layers, folder, 50)
    elif load_file == 'Y':
        conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias = load_model(path)
        conv_activation_string, fc_activation_string, num_conv_pool_layers = get_preset_values(path)
        conv_activation_fn, _ = set_activation_function(conv_activation_string)
        fc_activation_fn, _ = set_activation_function(fc_activation_string)
        print("Testing data on trained model...")
        for i, (image_batch, label_batch) in enumerate(zip(test_batch_images, test_batch_labels)):
            print(f"Processing batch {i + 1}...")
            test_network(image_batch, label_batch, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights,
                         output_bias, conv_activation_fn, fc_activation_fn, num_conv_pool_layers, None, None)


if __name__ == "__main__":
    learning_rates = config.learning_rates
    learning_rates_str = config.learning_rates_str
    test_num = config.test_num
    lr_index = config.lr_index

    # tanh, sigmoid, relu, leaky_relu
    activation_func = "leaky_relu"

    '''
        for x in range(5):
        for i in range(len(learning_rates)):
            print("######################################################################################################################")
            print("Test: ", x+1)
            print("Learning Rate: ", learning_rates[i])
            print("Activation Function: ", activation_func)
            print("######################################################################################################################")
            # tanh, sigmoid, relu, leaky_relu
            main(activation_func, learning_rates[i], learning_rates_str[i], (x+1))
    '''


    print("Test: ", test_num)
    print("Learning Rate: ", learning_rates[lr_index])
    print("Activation Function: ", activation_func)
    main(activation_func, learning_rates[lr_index], learning_rates_str[lr_index], test_num)
