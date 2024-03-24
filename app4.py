import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer

# Load saved model


# Define a custom object scope to include the required custom layers
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the model with the custom object scope
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('rsdmodel.h5')

# Define parameters
img_size = (224, 224)

# Define class labels (you can adjust this based on your dataset)
class_names = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 'speed limit 100', 'speed limit 120', 'stop sign',]


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index1.html', prediction_text='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index1.html', prediction_text='No selected file')

    if file:
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

         # Update with actual class names

        return render_template('index1.html', prediction_text=f'Predicted Class: {class_names[predicted_class]}')

if __name__ == '__main__':
    app.run(debug=True)