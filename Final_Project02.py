#Final Project 02
#Full Name: Kirat Kaur
#Student Number: 501125524

#Importing required packages 
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__) #To check the presence of tenserflow in spyder 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

## Step 1
#As mentioned in project file 
Imagesize = (500, 500)
batch = 32

#Deriving paths for Data 
train_data_file = 'C:\\Users\\user 123\\OneDrive\\Documents\\GitHub\\Final-Project-02\\Data\\train'
validation_data_file = 'C:\\Users\\user 123\\OneDrive\\Documents\\GitHub\\Final-Project-02\\Data\\valid'
test_data_file = 'C:\\Users\\user 123\\OneDrive\\Documents\\GitHub\\Final-Project-02\\Data\\test'

# Data Augmentation
train_data = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.3, 
    zoom_range = 0.2, 
    horizontal_flip = True)
vali_rescale = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

#Train and validation generator Using Keras
Train_generator = train_data.flow_from_directory(train_data_file, target_size = Imagesize, batch_size = batch, class_mode = 'categorical')
Val_generator = vali_rescale.flow_from_directory(validation_data_file, target_size = Imagesize, batch_size = batch, class_mode = 'categorical')
Test_generator = test_data.flow_from_directory(test_data_file, target_size = Imagesize, batch_size = batch, class_mode = 'categorical')

## Step 2
model = tf.keras.Sequential([

# Convolutional Layer & MaxPooling #1
    Conv2D(32, (3,3), (1,1), activation = 'relu', input_shape=(500,500,3)),
    MaxPooling2D(pool_size=(2,2)),
# Convolutional Layer & MaxPooling #2
    Conv2D(64, (3,3), (1,1), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
# Convolutional Layer & MaxPooling #3  
    Conv2D(64, (3,3), (1,1), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
# Convolutional Layer & MaxPooling #4
    Conv2D(128, (3,3), (1,1), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
 
#flatten layer 
    Flatten(),
     
# Dense layer and Dropout
    Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.02)),
    Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.6),
    Dense(3, activation='softmax')
    ])


## Step 3 will come backj to this after i have evaluated the model in step 4
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer= optimizer, loss= 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

## Step 4

earlystop = EarlyStopping(monitor = 'val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_v3.keras', monitor='val_loss', save_best_only=True)
history = model.fit(Train_generator, Validation_data=Val_generator, epochs=25, batch_size= batch, callbacks=[checkpoint, earlystop])

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()

plt.show()


## Step 5: Testing the model with test images
from tensorflow.keras.preprocessing import image
import numpy as np

# Define the paths for the test images
test_images = {
    "test_crack": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\crack\\test_crack.jpg",
    "test_missinghead": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\missing-head\\test_missinghead.jpg",
    "test_paintoff": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\paint-off\\test_paintoff.jpg"
}

# Load the trained model (ensure that the model has been saved as 'best_model_v3.keras')
model = tf.keras.models.load_model('best_model_v3.keras')

# Function to preprocess the image and predict its class
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=Imagesize)  # Resize to the model input size (500, 500)
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (scale to [0, 1])
    
    # Make predictions
    prediction = model.predict(img_array)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Map the predicted class index to class labels
    class_labels = ['crack', 'missing-head', 'paint-off']
    predicted_class = class_labels[predicted_class_index]
    
    # Print the prediction result
    print(f"Predicted Class for {image_path}: {predicted_class} (Probability: {prediction[0][predicted_class_index]:.4f})")
    
    # Optionally, display the image (for visual inspection)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Predict the class for each test image
for key, image_path in test_images.items():
    predict_image(image_path)

