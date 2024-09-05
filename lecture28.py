# Import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the class labels
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialize the figure and subplots
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Iterate through the first 10 images
for i, ax in enumerate(axes.flat):
    # Select the image and label
    image, label = x_train[i], y_train[i]

    # Display the image
    ax.imshow(image, cmap='gray')

    # Set the title with the class label
    ax.set_title(f"{class_labels[label.item()]}")
    ax.axis('off')

# Display the figure
plt.show()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices / one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

y_train[0]

  # Compile the model
model.compile(optimizer= SGD(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_test, y_test))
# # Define the CNN model
# model = Sequential()
# # Conv2D layer with 32 filters, kernel size 3x3, input shape (32, 32, 3)
# # Input size: 32x32x3, Kernel size: 3x3, Number of kernels: 32, Output size: 30x30x32. [input_size-kernel]+1
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# # MaxPooling2D layer with pool size 2x2
# # Output size: 15x15x32
# model.add(MaxPooling2D((2, 2)))

# #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# # Conv2D layer with 64 filters, kernel size 3x3
# # Input size: 15x15x32, Kernel size: 3x3, Number of kernels: 64, Output size: 13x13x64
# model.add(Conv2D(128, (3, 3), activation='relu'))
# # MaxPooling2D layer with pool size 2x2
# # Output size: 6x6x64
# model.add(MaxPooling2D((2, 2)))
# # Conv2D layer with 64 filters, kernel size 3x3
# # Input size: 6x6x64, Kernel size: 3x3, Number of kernels: 64, Output size: 4x4x64
# model.add(Conv2D(128, (3, 3), activation='relu'))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))


# # Flatten layer
# # Output size: 1024
# model.add(Flatten())
# # Dense layer with 64 units
# # Input size: 1024, Output size: 64
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# # Dense layer with 10 units (output layer)
# # Input size: 64, Output size: 10

# model.add(Dense(10, activation='softmax'))

# # TEST ACCURACY: 71%+

# Define the CNN model
model = Sequential()
# Conv2D layer with 32 filters, kernel size 3x3, input shape (32, 32, 3)
# Input size: 32x32x3, Kernel size: 3x3, Number of kernels: 32, Output size: 30x30x32. [input_size-kernel]+1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# MaxPooling2D layer with pool size 2x2
# Output size: 15x15x32
model.add(MaxPooling2D((2, 2)))

#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# Conv2D layer with 64 filters, kernel size 3x3
# Input size: 15x15x32, Kernel size: 3x3, Number of kernels: 64, Output size: 13x13x64
model.add(Conv2D(128, (3, 3), activation='relu'))
# MaxPooling2D layer with pool size 2x2
# Output size: 6x6x64
model.add(MaxPooling2D((2, 2)))
# Conv2D layer with 64 filters, kernel size 3x3
# Input size: 6x6x64, Kernel size: 3x3, Number of kernels: 64, Output size: 4x4x64
model.add(Conv2D(128, (3, 3), activation='relu'))



# Flatten layer
# Output size: 1024
model.add(Flatten())
# Dense layer with 64 units
# Input size: 1024, Output size: 64
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
# Dense layer with 10 units (output layer)
# Input size: 64, Output size: 10

model.add(Dense(10, activation='softmax'))

# 73+ accuracy

#compiling
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#other optimizer SGD, RMSprop, Adadelta, Adagrad, Adamax etc
# other loss functions are mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, binary_crossentropy

#training 
model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)

pred = model.predict(x_test)


num_images_to_display = 20
num_columns = 4
num_rows = (num_images_to_display + num_columns - 1) // num_columns

fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    if i < num_images_to_display:
        ax.imshow(x_test[i])
        actual_label = class_labels[np.argmax(y_test[i])]
        predicted_label = class_labels[np.argmax(pred[i])]
        ax.set_title(f"Actual: {actual_label}, Predicted: {predicted_label}")
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
