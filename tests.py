import unittest
from main2 import *


class TestCNNFunctions(unittest.TestCase):
    def test_relu(self):
        x = np.array([[-1, 2], [0, -3]])
        expected = np.array([[0, 2], [0, 0]])
        result = relu(x)
        np.testing.assert_array_equal(result, expected)

    def test_derivative_relu(self):
        x = np.array([[-1, 2], [0, -3]])
        expected = np.array([[0, 1], [0, 0]])
        result = derivative_relu(x)
        np.testing.assert_array_equal(result, expected)

    def test_tanh(self):
        x = np.array([[-1, 0], [1, 2]])
        result = tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_equal(result, expected)

    def test_derivative_tanh(self):
        x = np.array([[-1, 0], [1, 2]])
        expected = 1 - np.tanh(x) ** 2
        result = derivative_tanh(x)
        np.testing.assert_array_equal(result, expected)

    def test_sigmoid(self):
        x = np.array([0, 2])
        result = sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_equal(result, expected)

    def test_derivative_sigmoid(self):
        x = np.array([[-1, 0], [1, 2]])
        sig = sigmoid(x)
        expected = sig * (1 - sig)
        result = derivative_sigmoid(x)
        np.testing.assert_array_equal(result, expected)

    def test_one_hot_encode(self):
        data = np.array([0, 1, 2])
        num_classes = 3
        expected = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        result = one_hot_encode(data, num_classes)
        np.testing.assert_array_equal(result, expected)

    def test_remap_classes(self):
        neither_label = 'N'
        labels = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        classes_to_train = [2, 3, 4, 5, 6, 7]
        expected = np.array([0, 0, 1, 2, 3, 4, 5, 0])
        result = remap_classes(labels, neither_label, classes_to_train)
        np.testing.assert_array_equal(result, expected)

    def test_normalize_images(self):
        images = np.array([[0, 128, 255], [64, 192, 32]])
        expected = np.array([[0, 128 / 255, 1], [64 / 255, 192 / 255, 32 / 255]])
        result = normalize_images(images)
        np.testing.assert_almost_equal(result, expected, decimal=6)


    def test_convolution(self):
        image = np.random.rand(5, 5, 3)
        kernel = np.random.rand(3, 3, 3, 8)
        output = convolution(image, kernel, stride=1, padding=1)
        self.assertEqual(output.shape, (5, 5, 8))

    def test_max_pooling(self):
        # Create a sample 4x4 image with 3 channels
        image = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                          [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
                          [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]],
                          [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]]])

        # Expected output after 2x2 max pooling with stride=2
        expected_output = np.array([[[16, 17, 18], [22, 23, 24]],
                                    [[40, 41, 42], [46, 47, 48]]])

        # Call the max pooling function
        output = max_pooling(image, size=2, stride=2)

        # Test the output
        np.testing.assert_array_equal(output, expected_output)

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096])
        output = softmax(x)
        assert np.allclose(output, expected_output)

    def test_softmax_derivative(self):
        output = np.array([0.1, 0.7, 0.2])
        labels = np.array([0, 1, 0])
        expected_output = np.array([0.1, -0.3, 0.2])
        grad = softmax_derivative(output, labels)
        assert np.allclose(grad, expected_output)

    def test_fully_connected_operation(self):
        x = np.array([0.5, 0.2])
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        bias = np.array([0.1, 0.2])
        expected_output = np.array([0.21, 0.38])  # Corrected values
        output = fully_connected_operation(x, weights, bias)
        assert np.allclose(output, expected_output)

    def test_initialize_weights(self):
        conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias = initialize_weights()
        assert conv_kernels is not None
        assert fully_connected_weights.shape == (1280, 20)
        print(output_weights.shape)
        assert output_weights.shape == (20, 7)

    def test_set_activation_function(self):
        activation_fn, activation_backward_fn = set_activation_function('relu')
        assert activation_fn == relu
        assert activation_backward_fn == relu_backward

    def test_cross_entropy_loss(self):
        predictions = np.array([0.2, 0.3, 0.5])
        labels = np.array([0, 0, 1])
        expected_loss = 0.6931471805599453
        loss = cross_entropy_loss(predictions, labels)
        assert np.isclose(loss, expected_loss)

    def test_update_parameters(self):
        params = [np.array([0.5, 0.2]), np.array([0.1])]
        grads = [np.array([0.1, 0.05]), np.array([0.01])]
        learning_rate = 0.1
        update_parameters(params, grads, learning_rate)
        expected_params = [np.array([0.49, 0.195]), np.array([0.099])]
        for param, expected_param in zip(params, expected_params):
            assert np.allclose(param, expected_param)

    def test_fully_connected_backward(self):
        output_gradient = np.array([0.1, 0.2])
        fully_connected_input = np.array([0.5, 0.2])
        fully_connected_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        grad_input, grad_weights, grad_biases = fully_connected_backward(output_gradient, fully_connected_input,
                                                                         fully_connected_weights)
        expected_grad_weights = np.array([[0.05, 0.1], [0.02, 0.04]])
        expected_grad_biases = np.array([0.1, 0.2])
        assert np.allclose(grad_weights, expected_grad_weights)
        assert np.allclose(grad_biases, expected_grad_biases)

    def test_backward_pass(self):
        output = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
        label = np.array([[0, 0, 1, 0, 0, 0, 0]])
        fully_connected_weights = np.random.randn(1792, 128)
        output_weights = np.random.randn(128, 7)
        activation_fn, activation_backward_fn = set_activation_function('relu')
        forward_pass_params = [
            np.random.randn(1, 128),
            np.random.randn(1, 128),
            np.random.randn(1, 1792)]
        grads, loss = backward_pass(output, label, fully_connected_weights, output_weights, activation_backward_fn,
                                    forward_pass_params)
        assert loss > 0
        assert len(grads) == 4

    def test_forward_pass(self):
        image = np.random.randn(32, 32, 3)
        conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias = initialize_weights()
        fc_activation_fn, _ = set_activation_function('relu')
        conv_activation_fn, _ = set_activation_function('leaky_relu')
        output, _ = forward_pass(image, conv_kernels, fully_connected_weights, fully_connected_bias, output_weights,
                                 output_bias, conv_activation_fn, fc_activation_fn, 1)
        assert output.shape == (7,)

    def test_train_network(self):
        train_images = np.random.randn(10, 32, 32, 3)
        train_labels = np.random.randint(0, 7, size=(10, 1))
        train_labels = one_hot_encode(train_labels, 7)
        conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias, conv_activation_fn, fc_activation_fn = train_network(
            train_images, train_labels, 7, num_epochs=1)

    def test_check_weight_initialization(self):
        conv_kernels, fully_connected_weights, fully_connected_bias, output_weights, output_bias = initialize_weights(2,
                                                                                                                      2,
                                                                                                                      1,
                                                                                                                      .1)
        print("Convolutional Kernels:\n", conv_kernels)
        print("\nFully Connected Weights:\n", fully_connected_weights)
        print("\nFully Connected Bias:\n", fully_connected_bias)
        print("\nOutput Weights:\n", output_weights)
        print("\nOutput Bias:\n", output_bias)

    def test_convolution2(self):
        # Test input (height, width, channels)
        image = np.random.rand(7, 7, 3)  # Example 7x7 RGB image
        kernel = np.random.rand(3, 3, 3, 2)  # Example 3x3 kernels, 2 output filters

        # Test without padding and stride
        output = convolution(image, kernel, stride=1, padding=0)
        assert output.shape == (5, 5, 2), f"Expected output shape (5, 5, 3, 2), got {output.shape}"

        # Test with padding
        output_with_padding = convolution(image, kernel, stride=1, padding=1)
        assert output_with_padding.shape == (
        7, 7, 2), f"Expected output shape (7, 7, 3, 2), got {output_with_padding.shape}"

        # Test stride
        output_with_stride = convolution(image, kernel, stride=2, padding=0)
        assert output_with_stride.shape == (
        3, 3, 2), f"Expected output shape (3, 3, 3, 2), got {output_with_stride.shape}"

        print("Convolution tests passed.")

    def test_softmax2(self):
        # Test input array
        x = np.array([2.0, 1.0, 0.1])

        output = softmax(x)
        assert np.all(output >= 0), "Softmax output should be non-negative."
        assert np.isclose(np.sum(output), 1), f"Softmax output should sum to 1, got {np.sum(output)}"

        print("Softmax tests passed.")

    def test_softmax_derivative2(self):
        # Test softmax output and labels
        output = np.array([0.7, 0.2, 0.1])  # Example softmax output
        labels = np.array([1, 0, 0])  # Example one-hot encoded labels

        gradient = softmax_derivative(output, labels)
        expected_gradient = output - labels
        assert np.allclose(gradient, expected_gradient), f"Expected gradient {expected_gradient}, got {gradient}"

        print("Softmax derivative tests passed.")

if __name__ == "__main__":
    unittest.main()
