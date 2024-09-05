# Autoencoders are a particular type of neural network, just like classifiers. Autoencoders are similar to classifiers in the sense that they compress data. However, where classifiers condense all the data of an image into a single label, autoencoders compress the data into a latent vector, often denoted  z  in literature, with the goal of preserving the opportunity to recreate the exact same image in the future. Because autoencoders learn representations instead of labels, autoencoders belong to representation learning, a subfield of machine learning, but not necessarily deep learning.

# While recreating the same data from a compressed version might seem like an impossible task. However, you can actually do the same. You probably have no difficulty memorizing the following sequence:

# 1,3,5,7,9,11,13,15,17,19,21,23,25,27... 

# I bet you haven't looked at every item, but you can still write down the sequence perfectly because you recognized a pattern: all uneven numbers, starting from 1.

# This is what autoencoders do: they find patterns in data.

# Architecture
# Autoencoders consist of two networks:

# Encoder
# Decoder
# The goal of the encoder is to compress an image, video, or any piece of data that can be represented as a tensor, into a latent vector. The decoder does, as you might have guessed, the exact opposite.

# To maximize performance, minimize the loss that is, encoders and decoders are typically symmetrical together. Naturally, the input size is equal to the output size of an autoencoder.

# Autoencoders always have less input neurons in the middle layer than in the input and output layer. This is called the bottleneck. If it weren't for this bottleneck, the autoencoders could just copy this data over from the input to the output layer without compressing it.

# Training
# Encoders and decoders can be trained separately, but usually they are trained in one go. In order to do so, one stacks the coders together in one stacked autoencoder.

# If one desires to train autoencoders separately, one starts by using the first hidden layer, discaring every other layer, except for the input and output layers of course. He uses the original training data at this point. Next, he uses the latent vector  z  learnt by this mini-autoencoder and trains another autoencoder in the same way, treating the latent vectors as original data. Once the desired depth is reached, one can stack all output layers, which provided the latent vectors, together in a sinle encoder. This approach is not used in practise a lot, but literature might refer to it as greedy layerwise training so it's good to know what it means.

# Appliciations
# While the phase "finding patterns" might not seem very interesting, there are a lot of exciting applications of autoencoders. We will look at three of those today:

# Dense autoencoder: compressing data.
# Convolutional autoencoder: a building block of DCGANs, self-supervised learning.
# Denoising autoencoder: removing noise from poor training data.
# While all of these applications use pattern finding, they have different use cases making autoencoders one of the most exciting topics of machine learning.

from tensorflow import keras

# Load the MNIST dataset from Keras
(x_train,_),(x_test,_)=keras.datasets.mnist.load_data()
# Normalize the training data to be in the range [0, 1]
x_train=x_train/255.0
# Normalize the test data to be in the range [0, 1]
x_test=x_test/255.0

# A simple autoencoder
# Let's start by looking at the simplest possible autoencoder.

# The encoder is a sequential neural network with 28×28 input neurons, 100 neurons in the second layer and 30 in the third. The third layer is called the "bottleneck". Feel free to play around with this variable to see how it affects results.

# Create the encoder model
encoder=keras.models.Sequential([
    # Flatten the input images (28x28) into a 1D array of 784 elements
    keras.layers.Flatten(input_shape=[28,28]),
    # Add a dense layer with 100 neurons and ReLU activation function
    keras.layers.Dense(100,activation="relu"),
    # Add another dense layer with 30 neurons and ReLU activation function
    keras.layers.Dense(30,activation="relu")
])

#The decoder is the same, but in opposite order. Note that keras needs to know the input shape at this point. The input shape of the decoder is the shape of $z$, also called `zDim` as you will see later on.

decoder=keras.models.Sequential([
    # Add a dense layer with 100 neurons and ReLU activation function, input shape is 30 (output from encoder)
    keras.layers.Dense(100,activation=100,input_shape=[30]),
    # Add a dense layer with 784 neurons (28*28) and sigmoid activation function to output pixel values in range [0, 1]
    keras.layers.Dense(28*28,activation="sigmoid"),
    # Reshape the output into the original image shape (28x28)
    keras.layers.Reshape([28,28])
])

# Stack the encoder and decoder models to create the autoencoder
stacked_autoencoder=keras.models.Sequential([encoder,decoder])

#Note that we use binary cross entropy loss in stead of categorical cross entropy. The reason for that is because we are not classifying latent vectors to belong to a particular class, we do not even have classes!, but rather are trying to predict whether a pixel should be activated or not.

# Compile the stacked autoencoder model
stacked_autoencoder.compile(
    # Use binary cross-entropy loss function for binary input data (normalized pixel values)
    loss="binary_crossentropy",
    # Use the Adam optimizer for efficient training
    optimizer='adam'
)

# Train the stacked autoencoder model
history=stacked_autoencoder.fit(
    # Input and target data are both x_train (input images) for reconstruction
    x_train,x_train,
    epochs=10,
    # Use x_test for validation during training
    validation_data=[x_test,x_test]
)

# # Set the figure size for the plot
# figsize(20,5)
# # Iterate over a range of 8 examples from the test set
# for i in range(8):
#     #plot the original image from the test set
#     subplot(2,8,i+1)
#     # Make a prediction using the stacked autoencoder on the current test image
#     pred=stacked_autoencoder.predict(x_test[i].reshape((1,28,28)))
#     # Display the original image
#     imshow(x_test[i],cmap='binary')

#     # Plot the reconstructed image by the stacked autoencoder
#     subplot(2,8,i+8+1)
#     # Display the reconstructed image
#     imshow(pred.reshape((28,28)),cmap="binary")

import matplotlib.pyplot as plt

# Set the figure size for the plot
plt.figure(figsize=(20, 5))

# Iterate over a range of 8 examples from the test set
for i in range(8):
    # Plot the original image from the test set
    plt.subplot(2, 8, i+1)
    # Make a prediction using the stacked autoencoder on the current test image
    pred = stacked_autoencoder.predict(x_test[i].reshape((1, 28, 28)))
    # Display the original image
    plt.imshow(x_test[i], cmap='binary')
    plt.axis('off')  # Optional: turn off axis for a cleaner look

    # Plot the reconstructed image by the stacked autoencoder
    plt.subplot(2, 8, i+8+1)
    # Display the reconstructed image
    plt.imshow(pred.reshape((28, 28)), cmap="binary")
    plt.axis('off')  # Optional: turn off axis for a cleaner look

# Show the plot
plt.show()


# # Set the index i to choose a specific example from the test set
# i = 0  # Change this number to select a different example

# # Set the figure size for the plot
# figsize(10, 5)

# # Plot the original image
# subplot(1, 3, 1)
# imshow(x_test[i], cmap="binary")

# # Plot the latent vector representation obtained from the encoder
# subplot(1, 3, 2)
# # Predict the latent vector representation of the selected test image
# latent_vector = encoder.predict(x_test[i].reshape((1, 28, 28)))
# imshow(latent_vector, cmap="binary")

# # Plot the reconstructed image from the latent vector using the decoder
# subplot(1, 3, 3)
# # Reconstruct the image from the latent vector representation
# pred = decoder.predict(latent_vector)
# imshow(pred.reshape((28, 28)), cmap="binary")

# 30 / (28 * 28), 1 - 30 / (28 * 28)

# Calculate the sparsity constraints
sparsity_low = 30 / (28 * 28)  # Lower bound for sparsity constraint
sparsity_high = 1 - 30 / (28 * 28)  # Upper bound for sparsity constraint

# Print the calculated values
print(sparsity_low, sparsity_high)


# Define the encoder model using convolutional layers
encoder = keras.models.Sequential([
    # Reshape the input into a 28x28x1 image (grayscale)
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    # First convolutional layer with 16 filters, each of size 3x3, ReLU activation, and padding to maintain size
    keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    # Max pooling layer with pool size 2x2 to downsample the spatial dimensions
    keras.layers.MaxPool2D(pool_size=2),
    # Second convolutional layer with 32 filters, each of size 3x3, ReLU activation, and padding
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    # Max pooling layer
    keras.layers.MaxPool2D(pool_size=2),
    # Third convolutional layer with 64 filters, each of size 3x3, ReLU activation, and padding
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    # Max pooling layer
    keras.layers.MaxPool2D(pool_size=2)
])


# Predict the encoded representation of a single test image and print its shape
encoder.predict(x_test[0].reshape((1, 28, 28))).shape


# Define the decoder model using convolutional transpose layers
decoder = keras.models.Sequential([
    # Convolutional transpose layer with 32 filters, each of size 3x3, stride 2, ReLU activation, and valid padding
    keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding="valid",
                                 activation="relu",
                                 input_shape=[3, 3, 64]),
    # Convolutional transpose layer with 16 filters, each of size 3x3, stride 2, ReLU activation, and same padding
    keras.layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding="same",
                                 activation="relu"),
    # Convolutional transpose layer with 1 filter, size 3x3, stride 2, sigmoid activation, and same padding
    keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), strides=2, padding="same",
                                 activation="sigmoid"),
    # Reshape the output into the original image shape (28x28)
    keras.layers.Reshape([28, 28])
])


# Stack the encoder and decoder models to create the stacked autoencoder
stacked_autoencoder = keras.models.Sequential([encoder, decoder])

# Compile the stacked autoencoder model
stacked_autoencoder.compile(
    # Use binary cross-entropy loss function for binary input data (normalized pixel values)
    loss="binary_crossentropy",
    # Use the Adam optimizer for efficient training
    optimizer='adam'
)

# Train the stacked autoencoder model on the training data
history = stacked_autoencoder.fit(
    # Input and target data are both x_train (input images) for reconstruction
    x_train, x_train,
    # Train for 10 epochs
    epochs=10,
    # Use x_test for validation during training
    validation_data=[x_test, x_test]
)

# Set the figure size for the plot
figsize(20, 5)

# Iterate over 8 examples from the test set
for i in range(8):
    # Plot the original image from the test set
    subplot(2, 8, i+1)
    # Make a prediction using the stacked autoencoder on the current test image
    pred = stacked_autoencoder.predict(x_test[i].reshape((1, 28, 28)))
    # Display the original image
    imshow(x_test[i], cmap="binary")

    # Plot the reconstructed image by the stacked autoencoder
    subplot(2, 8, i+8+1)
    # Display the reconstructed image
    imshow(pred.reshape((28, 28)), cmap="binary")

# Set the figure size for the plot
figsize(15, 15)

# Iterate over all filters in the last convolutional layer of the encoder
for i in range(8 * 8):
    # Plot each filter as a subplot in an 8x8 grid
    subplot(8, 8, i+1)
    # Display the weights (filters) of the convolutional layer
    imshow(encoder.layers[-2].weights[0][:, :, 0, i])

# Visually not very pleasing, but proven to be effective as shown in the previous figure.

# 3×3×64=576  is still less than  28×28=784 , thus creating a bottleneck, but much less compressed than the dense encoder making convolutional encoders less suitable for comporession. But thanks to their convolutional layers, they are great to use in cases where you want your autoencoder to find visual patterns in your data.

# Denoising autoencoder
# The last application of autoencoders we look at today are denoising autoencoders. You probably have no difficulty classifying the images below as 7's.

import numpy as py
# Set the figure size for the plot
figsize(5, 10)

# Plot the original image from the test set
subplot(1, 2, 1)
imshow(x_test[0], cmap="binary")

# Plot the noisy version of the original image
subplot(1, 2, 2)
# Generate random noise and add it to the original image
noise = np.random.random((28, 28)) / 4
imshow(x_test[0] + noise, cmap="binary")

# Define the encoder model using dense (fully connected) layers
encoder = keras.models.Sequential([
    # Flatten the input images (28x28) into a 1D array of 784 elements
    keras.layers.Flatten(input_shape=[28, 28]),
    # Add a dense layer with 100 neurons and ReLU activation function
    keras.layers.Dense(100, activation="relu"),
    # Add another dense layer with 100 neurons and ReLU activation function
    keras.layers.Dense(100, activation="relu"),
    # Add a dense layer with 30 neurons and ReLU activation function to generate the latent space representation
    keras.layers.Dense(30, activation="relu")
])

# Define the decoder model using dense (fully connected) layers
decoder = keras.models.Sequential([
    # Add a dense layer with 100 neurons and ReLU activation function, input shape is 30 (latent space representation)
    keras.layers.Dense(100, activation="relu", input_shape=[30]),
    # Add another dense layer with 100 neurons and ReLU activation function
    keras.layers.Dense(100, activation="relu"),
    # Add a dense layer with 784 neurons (28*28) and sigmoid activation function to output pixel values in range [0, 1]
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    # Reshape the output into the original image shape (28x28)
    keras.layers.Reshape([28, 28])
])

# Combine the encoder and decoder models to create the stacked autoencoder
stacked_autoencoder = keras.models.Sequential([encoder, decoder])

# Compile the stacked autoencoder model
stacked_autoencoder.compile(
    # Use binary cross-entropy loss function for binary input data (normalized pixel values)
    loss="binary_crossentropy",
    # Use the Adam optimizer for efficient training
    optimizer='adam'
)

# Add random noise to the training and test data
x_train_noise = x_train + ((np.random.random(x_train.shape)) / 4)
x_test_noise = x_test + ((np.random.random(x_test.shape)) / 4)
# Display an example of the noisy training data
imshow(x_train_noise[0], cmap="binary")

# Train the stacked autoencoder model on the noisy training data
history = stacked_autoencoder.fit(
    # Input is the noisy training data (x_train_noise), target is the original clean training data (x_train)
    x_train_noise, x_train,
    # Train for 10 epochs
    epochs=10,
    # Use noisy test data (x_test_noise) for validation during training, with original clean test data (x_test) as target
    validation_data=[x_test_noise, x_test]
)

# Set the figure size for the plot
figsize(20, 5)

# Iterate over 8 examples from the noisy test set
for i in range(8):
    # Plot the noisy version of the original image from the test set
    subplot(2, 8, i+1)
    imshow(x_test_noise[i], cmap="binary")

    # Plot the reconstructed image by the stacked autoencoder using the noisy input
    subplot(2, 8, i+8+1)
    # Make a prediction using the stacked autoencoder on the current noisy test image
    pred = stacked_autoencoder.predict(x_test_noise[i].reshape((1, 28, 28)))
    # Display the reconstructed image
    imshow(pred.reshape((28, 28)), cmap="binary")

