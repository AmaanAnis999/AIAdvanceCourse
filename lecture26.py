import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

# Define the class names
class_names=['airplanes','automobiles','bird','cat','deer','dog','frog','horse','ship','truck']

# Build a simple ANN model
model=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10,activation='softmax'),
])

# my practice
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),  # MaxPooling should come before Flatten
    layers.Flatten(),  # Flatten the pooled feature map
    layers.Dense(512, activation='relu'),  # Dense layer with 512 units
    layers.Dense(256, activation='relu'),  # Dense layer with 256 units
    layers.Dense(10, activation='softmax'),  # Output layer with 10 units (for 10 classes)
])

#2nd try using chatgpt
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer for regularization
    layers.Dense(10, activation='softmax')
])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training model
history=model.fit(x_train,y_train,epochs=10,
                  validation_data=(x_test,y_test))

#evaluate the model
test_loss,test_acc=model.evaluate(x_test,y_test,verbose=2)
print(f'\nTest accuracy: {test_acc}')

#plotting training and validation accuracy
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history[val_accuracy],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.show()

#predicting the first 5 images in the test set
predictions=model.predict(x_test[:5])

#display the predictions 
for i in range(5):
    plt.figure()
    plt.imshow(x_test[i])
    plt.title(f"Predicted:{class_names[predictions[i].argmax()]}")
    plt.show()

#train the model to achieve the highest possible accuracy as possible

# Test accuracy: 0.722599983215332

#end
