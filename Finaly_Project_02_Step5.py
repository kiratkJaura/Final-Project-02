# Importing necessary libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

print(tf.__version__)  # Ensures TensorFlow is installed and available

# Defining Image size
Imagesize = (500, 500)

# Paths for the test images
test_images = {
    "test_crack": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\crack\\test_crack.jpg",
    "test_missinghead": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\missing-head\\test_missinghead.jpg",
    "test_paintoff": "C:\\Users\\user 123\\OneDrive\\Desktop\\AER 850\\Project 2 Data\\Data\\test\\paint-off\\test_paintoff.jpg"
}

# Loading the trained model
model = tf.keras.models.load_model('best_model_v3.keras')

# Preprocessing the image and predicting its class
def predict_image(image_path):
    # Loading and preprocessing the images
    img = image.load_img(image_path, target_size=Imagesize)  # Resize to model input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    # Predicting the class probabilities
    prediction = model.predict(img_array)

    # Class labels
    class_labels = ['crack', 'missing-head', 'paint-off']

    # Map predictions to the corresponding classes
    for i, label in enumerate(class_labels):
        print(f"Probability for {label}: {prediction[0][i]:.4f}")

    # Get the predicted class with the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    # Display the predicted class and its probability
    print(f"\nPredicted Class for {image_path}: {predicted_class} (Probability: {prediction[0][predicted_class_index]:.4f})\n")

    # Display the image for visual inspection
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}", y=1.08)  # Move title above the image
    plt.axis('off')

    # Adding probability text on the image
    plt.text(
        10, 450,  # Bottom-left placement
        f"Crack: {prediction[0][0]*100:.1f}%\nMissing Head: {prediction[0][1]*100:.1f}%\nPaint Off: {prediction[0][2]*100:.1f}%",
        fontsize=12, color='green', bbox=dict(facecolor='white', alpha=0.8)
    )

    # To save the output image with predictions
    plt.savefig(f"model_testing_{predicted_class}.png")  # Save with a unique name based on class
    plt.show()

# Predicting the class for each test image
for key, image_path in test_images.items():
    print(f"Testing {key}: {image_path}")
    predict_image(image_path)
