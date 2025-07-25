import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Download Bird dataset
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d umairshahpirzada/birds-20-species-image-classification

!unzip -q birds-20-species-image-classification.zip -d bird_data

# Step 1: Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

train_dir = "bird_data/train"
val_dir = "bird_data/valid"

# Step 2: Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Step 3: Load images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Step 4: Load ResNet50 base (no top classifier)
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model

# Step 5: Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Step 6: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Step 8: Evaluate
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc:.4f}")

model.save("bird_data.keras")

Bird_Model = tf.keras.models.load_model("bird_data.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(Bird_Model)

tflite_model = converter.convert()

tflite_model_name = 'BirdClassificationModel.tflite'
with open('BirdClassificationModel.tflite', 'wb') as f:
    f.write(tflite_model)

files.download('BirdClassificationModel.tflite')

import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def predict_bird_species(img_path, model, class_indices):
    img_array = preprocess_image(img_path)
    predictions = Bird_Model.predict(img_array)
    predicted_index = np.argmax(predictions)

    # Invert the class_indices dictionary to map index to class name
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_class = class_labels[predicted_index]

    # Display
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class

# Path to your test image
test_image_path = "bird.png"

# Use the function
predicted_species = predict_bird_species(test_image_path, model, train_generator.class_indices)
print("Predicted Bird Species:", predicted_species)