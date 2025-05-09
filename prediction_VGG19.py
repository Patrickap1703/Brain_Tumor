import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

# Set the standard output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Set image size (should match the size used in training)
IMAGE_SIZE = (224, 224)

def load_trained_model(model_path):
    return load_model(model_path)

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale the image


    prediction = model.predict(img_array)
    class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

if __name__ == "__main__":
    # Load the trained model
    model_path = 'brain_tumor_vgg19_model.h5'  # Make sure this path is correct
    model = load_trained_model(model_path)

    # Example usage
    image_path = r'G:\Brain tumor\Brain-Tumor-detection-with-VGG19-main\tumor_dataset\Testing\meningioma\Te-me_0011.jpg'  # Replace with the path to your image
    predicted_class, confidence = predict_image(image_path, model)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")