import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Preprocess the Dataset
print("Downloading and loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension (required for CNNs: (28, 28) -> (28, 28, 1))
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Build the Convolutional Neural Network (CNN)
print("Building the CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 output classes (digits 0-9)
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
print("Training the model...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Evaluate Accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

# 6. Save the model for future use
model.save('mnist_cnn_model.h5')
print("Model saved as 'mnist_cnn_model.h5'")

# 7. Make a Prediction and Visualize it
print("\nRunning a test prediction...")
image_index = 0 # You can change this to test different images
test_image = x_test[image_index]
true_label = y_test[image_index]

# Predict
predictions = model.predict(np.expand_dims(test_image, axis=0))
predicted_label = np.argmax(predictions[0])

# Plot the result
plt.imshow(test_image.squeeze(), cmap=plt.cm.binary)
plt.title(f"True Label: {true_label} | Predicted: {predicted_label}")
plt.axis('off')
plt.show()
