import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set image size (VGG16 expects 224x224 images)
IMAGE_SIZE = (224, 224)

# Set the paths to your dataset
train_dir = 'tumor_dataset/Training'
test_dir = 'tumor_dataset/Testing'

# Number of classes
num_classes = 4  # glioma, pituitary, meningioma, no tumor

# Load the VGG16 model without the top layers (include_top=False)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers to prevent them from updating during training
for layer in vgg_base.layers:
    layer.trainable = False

# Add custom fully connected layers for our dataset
x = Flatten()(vgg_base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=vgg_base.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Rescale pixel values between 0 and 1
    rotation_range=20,          # Randomly rotate images
    width_shift_range=0.2,      # Shift images horizontally
    height_shift_range=0.2,     # Shift images vertically
    shear_range=0.2,            # Shear transformation
    zoom_range=0.2,             # Zoom transformation
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest',        # Fill empty pixels after transformation
    validation_split=0.2        # Use 20% of training data for validation
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load train and validation data from the same directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Specify this is the training set
)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Specify this is the validation set
)
# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

import pickle
with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
model.save('brain_tumor_vgg16_model.h5')

#Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()