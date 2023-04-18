import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Define class names
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

# Print sample images with labels
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(class_names[y_train[i][0]])
plt.show()

# Define the model architecture
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Make predictions on test data
predictions = model.predict(x_test)

# Show sample test images with predicted labels
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i])
    ax.set_xticks([])
    ax.set_yticks([])
    predicted_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[y_test[i][0]]
    if predicted_label == true_label:
        ax.set_xlabel(predicted_label, color="green")
    else:
        ax.set_xlabel(predicted_label, color="red")
plt.show()
