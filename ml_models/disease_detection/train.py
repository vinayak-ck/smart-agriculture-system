import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 🔹 Paths
train_dir = '../../data/plant_disease/train'
test_dir = '../../data/plant_disease/test'

# 🔹 Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 🔹 Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 🔹 Class names
class_names = train_data.class_names
print("Classes:", class_names)

# 🔹 Normalize
normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# 🔹 CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# 🔹 Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🔹 Train
model.fit(train_data, epochs=5, validation_data=test_data)

# 🔹 Save model
model.save('cnn_model.h5')

# 🔹 Save labels
with open('labels.txt', 'w') as f:
    for label in class_names:
        f.write(label + '\n')

print("CNN Model saved!")