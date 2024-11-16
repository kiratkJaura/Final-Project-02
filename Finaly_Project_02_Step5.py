
## Step 5: Testing the model with test images
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__) #To check the presence of tenserflow in spyder 
model = tf.keras.Sequential
from tensorflow.keras.preprocessing import image
import numpy as np

Imagesize = (500, 500)
batch = 32

# Defining the paths for the test images
test_images = {
    "test_crack": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\crack\\test_crack.jpg",
    "test_missinghead": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\missing-head\\test_missinghead.jpg",
    "test_paintoff": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\paint-off\\test_paintoff.jpg"
}

# Loading the trained model (and ensuring that the model has been saved as 'best_model_v3.keras')
model = tf.keras.models.load_model('best_model_v3.keras')

# preprocessing the image and predicting its class
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=Imagesize)  # Resize to the model input size (500, 500)
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image (scale to [0, 1])
    
    # Making predictions
    prediction = model.predict(img_array)
    
    # Getting the highest probability of the class
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Mapping the predicted class index to class labels
    class_labels = ['crack', 'missing-head', 'paint-off']
    predicted_class = class_labels[predicted_class_index]
    
    # Predicted result
    print(f"Predicted Class for {image_path}: {predicted_class} (Probability: {prediction[0][predicted_class_index]:.4f})")
    
    # Displaying the image (for visual inspection)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Predicting the class for each test image
for key, image_path in test_images.items():
    predict_image(image_path)

