# General settings
load_file = 'N'  # 'Y' or 'N'
path = "tests/5_relu_1" # point to results folder

# Network variables
num_conv_pool_layers = 1
conv_activation = 'leaky_relu' # actyivation function in convolution layer
num_neurons = 50 # fully connected layer
num_epochs = 30
weight_scale = .05

# Learning settings
test_num = 1 # Must rename if you want to use same settings again
learning_rates = [.001, .0001, .00001]
learning_rates_str = ["001", "0001", "00001"]
lr_index = 0 # Choose from learning rates
learning_rate = learning_rates[lr_index] # DO NOT TOUCH
activ_str = 'relu' # tanh, sigmoid, relu, leaky_relu
test_name = f"test\\{test_num}_{activ_str}_{learning_rates_str[lr_index]}"

# Classification settings
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
fc_activation = activ_str  # tanh, sigmoid, relu, leaky_relu
classes_to_train = [2, 3, 4, 5, 6, 7] 
neither_label = 'N'  # 'Y' OR 'N'   This will use all classes from dataset, including non animal classes. It will classify anything that is not an animal as neither, resulting as a neither classification.

# Data limits
train_num = 30000 # Max: 30000
test_num = 6000 # Max: 6000
