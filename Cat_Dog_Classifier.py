import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing import image
import numpy as np

# Paths for training and validation datasets (adjust according to your structure)
train_dir = r'C:\Users\siyra\OneDrive\Desktop\train'
validation_dir = r'C:\Users\siyra\OneDrive\Desktop\validation'

# Function to check for corrupt images
def validate_images(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify that it is, in fact, an image
            except (IOError, UnidentifiedImageError):
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

# Validate images in train and validation directories
validate_images(train_dir)
validate_images(validation_dir)

# Data Preprocessing and Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Rescaling the pixel values between 0 and 1
    rotation_range=40,         # Data augmentation (rotation)
    width_shift_range=0.2,     # Horizontal shift
    height_shift_range=0.2,    # Vertical shift
    shear_range=0.2,           # Shearing
    zoom_range=0.2,            # Zooming
    horizontal_flip=True,      # Horizontal flip
    fill_mode='nearest'        # Filling in missing pixels
)

# Validation data only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load the images from directory and apply augmentation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),    # Resizing images to 150x150
    batch_size=32,
    class_mode='binary'        # Binary classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),    # Resizing validation images to 150x150
    batch_size=32,
    class_mode='binary'
)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Model compilation
model.compile(
    loss='binary_crossentropy',        # Loss function for binary classification
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),   # RMSProp optimizer
    metrics=['accuracy']               # Tracking accuracy
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,                # Adjust according to your dataset
    epochs=30,                          # Number of epochs
    validation_data=validation_generator,
    validation_steps=50
)

# Plot training and validation metrics
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.show()

# Optional: Uncomment to plot training history
# plot_training_history(history)

# Save the model after training
model.save("cat_dog_classifier_model.h5")

# Predict function for new images
def predict_image(model, img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Convert to a batch format
        
        img_array /= 255.0  # Normalize the image
        
        prediction = model.predict(img_array)
        if prediction[0] > 0.5:
            print("It's a dog!")
        else:
            print("It's a cat!")
    except UnidentifiedImageError:
        print(f"Error: Unable to load image {img_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Loop to predict multiple images
while True:
    # Ask the user for the image path
    test_image_path = input("Enter the path to the image (or type 'exit' to stop): ")
    
    # Exit the loop if the user types 'exit'
    if test_image_path.lower() == 'exit':
        print("Exiting prediction loop.")
        break
    
    # Predict the image
    predict_image(model, test_image_path)
