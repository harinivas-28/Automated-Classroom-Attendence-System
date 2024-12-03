import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load images and labels
images = np.load('images.npy')  # Shape should be (num_samples, height, width, channels)
labels_df = pd.read_csv('labels.csv')  # Assuming it has columns 'id' and 'count'

# Check if the number of images matches the number of labels
assert len(images) == len(labels_df), "Mismatch between number of images and labels"

# Prepare labels (assuming 'id' corresponds to the index of images)
labels = labels_df['count'].values

# Normalize images
images = images.astype('float32') / 255.0

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], images.shape[3])),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Output layer for count prediction
])

# Load the previously saved model
from tensorflow.keras.models import load_model
try:
    model = load_model('people_counting_model.h5')
except FileNotFoundError  as e:
    print("File Not Found")

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save model if needed
model.save('people_counting_model.h5')

# Example usage for prediction
def predict_count(image):
    image = np.expand_dims(image.astype('float32') / 255.0, axis=0)  # Normalize and add batch dimension
    return model.predict(image)[0][0]

# Load an example image for prediction (replace with your own image loading)
test_image = X_val[20]  # Just an example; replace with actual image loading logic.
predicted_count = predict_count(test_image)
plt.imshow(test_image)
predicted_count = predict_count(test_image)
print(f'Predicted number of people: {predicted_count:.2f}')
plt.show()





