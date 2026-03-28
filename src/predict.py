import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

# -------------------------------
# Load Model
# -------------------------------
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------------
# Predict Single Image
# -------------------------------
def predict_single_image(model, img_path, class_labels):
    img = preprocess_image(img_path)
    predictions = model.predict(img)

    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(predictions))

    return {
        "image": img_path,
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    }


# -------------------------------
# Predict Multiple Images (Batch)
# -------------------------------
def predict_folder(model, folder_path, class_labels):
    results = []

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            result = predict_single_image(model, img_path, class_labels)
            results.append(result)
        except:
            continue

    return results