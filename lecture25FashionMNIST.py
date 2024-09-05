###1. Load Data and Splot Data
#from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import layers , models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


#loading mnist fashion dataset
fashion_mnist=tf.keras.datasets.fashion_mnist

#split the data into test and train set
(X_train,Y_train),(X_test,X_train)=fashion_mnist.load_data()

#number of images to display 
n=10
#create a figure to display the images
plt.figure(figsize=(20,4))
# Loop through the first 'n' images
for i in range(n):
    #subplot within figure
    ax=plt.subplot(2,n,i+1)
    #display the original image
    plt.imshow(X_test[i].reshape(28,28))
    # Set colormap to grayscale
    plt.gray()

    # Hide x-axis and y-axis labels and ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#show the figure with images
plt.show()
#close the figure
plt.close

#shapes of original training data and labels
print("previous X_train shape: {} \nprevious Y_train shape:{}",format(X_train,Y_train.shape))

#reshape training and testing and testing data to a flat format 
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)

# MIN MAX SCALING
# Convert the data type of the images to float32
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

# Normalize the pixel values to a range between 0 and 1  # Zero is for Black  #1 for White
X_train/=255
X_test/=255

# Number of classes in the dataset
classes = 10
# Convert the labels to one-hot encoded format
Y_train = to_categorical(Y_train, classes)
Y_test = to_categorical(Y_test, classes)

# Print the shapes of the preprocessed training data and labels
print("New X_train shape: {} \nNew Y_train shape:{}".format(X_train.shape, Y_train.shape))

# Define the input size for each data sample (e.g., image pixels)
input_size = 784

# Specify the number of data samples to process in each batch
batch_size = 200

# Define the number of neurons in the first hidden layer
hidden1 = 400

# Define the number of neurons in the second hidden layer
hidden2 = 20

# Define the total number of classes/categories in the dataset
classes = 10

# Set the number of complete passes through the dataset during training
epochs = 10

### 4. Build the model ###

# Create a Sequential model, which allows us to build a neural network layer by layer
model = Sequential()

# Add the first hidden layer with 'hidden1' neurons, using ReLU activation function
# The 'input_dim' specifies the input size for this layer
model.add(Dense(hidden1, input_dim=input_size, activation='relu'))
# output = relu(dot(W, input) + bias)

# Add the second hidden layer with 'hidden2' neurons, also using ReLU activation function
model.add(Dense(hidden2, activation='relu'))


# Add the output layer with 'classes' neurons, using softmax activation function
# Softmax activation ensures that the output values represent probabilities of each class
model.add(Dense(classes, activation='softmax'))

### Compilation ###

# Compile the model by specifying the loss function, optimizer, and evaluation metrics
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='sgd')

# Display a summary of the model architecture, showing the layers and parameter counts
model.summary()
