Creating a Smart-Recycle-Bot using computer vision involves several steps, from setting up the environment to using a machine learning model to classify waste materials. Below is a complete Python program that implements this project. This example assumes you are using a pre-trained model like MobileNetV2 or similar for waste classification. For actual testing and deployment, you would customize this with your specific dataset and possibly train a new model.

Before running the program, make sure you have installed the necessary libraries:
```bash
pip install opencv-python opencv-python-headless opencv-contrib-python tensorflow numpy
```

Here is the complete Python program:

```python
import cv2
import numpy as np
import tensorflow as tf
import os

# Load your pre-trained model for waste classification
# This example uses a sample placeholder path for a model. You should replace it with your actual model path.
MODEL_PATH = 'path_to_your_model/saved_model'
model = tf.keras.models.load_model(MODEL_PATH)

# Define labels corresponding to the model's output, e.g., ['Paper', 'Plastic', 'Metal', 'Glass', 'Non-Recyclable']
 LABELS = ['Paper', 'Plastic', 'Metal', 'Glass', 'Non-Recyclable']

def preprocess_image(image_path):
    """
    Preprocess the image to feed into the model.
    - Reads an image from file
    - Resizes it to 224x224
    - Scales the pixel values
    :param image_path: Path of the image file
    :return: Preprocessed image
    """
    try:
      image = cv2.imread(image_path)
      image = cv2.resize(image, (224, 224))  # Resize to match model's expected input
      image = image / 255.0  # Scale pixel values from 0 to 1
      return np.expand_dims(image, axis=0)  # Add a batch dimension
    except Exception as e:
      print(f"Error in preprocessing image {image_path}: {e}")
      return None

def classify_waste(image_path):
    """
    Classify the waste type using the pre-trained model.
    :param image_path: Path of the image file
    :return: The predicted waste category
    """
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return "Error processing image."

    try:
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        label = LABELS[predicted_class]
        confidence = predictions[0][predicted_class]
        return f"Predicted: {label} with confidence: {confidence:.2f}"
    except Exception as e:
        print(f"Error in classification: {e}")
        return "Classification error."

def main():
    """
    Main function to execute the Smart-Recycle-Bot.
    """
    image_directory = 'path_to_images/'  # Directory containing sample images

    for filename in os.listdir(image_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check supported file formats
            image_path = os.path.join(image_directory, filename)
            result = classify_waste(image_path)
            print(f"Image: {filename}, Result: {result}")
        else:
            print(f"Unsupported file format or directory: {filename}")

if __name__ == "__main__":
    main()
```

### Key Points in the Program:
1. **Model Loading**: The `tensorflow.keras.models.load_model` function loads a saved model from a specified path. Be sure to replace `MODEL_PATH` with the actual path to your trained model.

2. **Image Preprocessing**: The `preprocess_image` function reads, resizes, and scales the image to be compatible with the input expectations of pre-trained models like MobileNetV2.

3. **Classification**: The `classify_waste` function predicts the class of the waste material using the model and returns the predicted label with confidence.

4. **Error Handling**: The program includes basic error handling during image reading, preprocessing, and model prediction stages.

5. **Execution**: The `main` function iterates over all image files in a specified directory and classifies them.

You’ll need to adapt this program to your actual dataset and trained model, especially adjusting the input processing and model path to fit the specifics of your situation. Note that for a real-world application, a more complex pipeline involving more robust data augmentation and potential retraining with a specific dataset might be necessary.