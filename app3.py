import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.layers import Layer

# Create a Flask app
app = Flask(__name__)

# Load the trained model
# or: from tf.keras.utils import custom_object_scope

# Define a custom object scope to include the required custom layers
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the model with the custom object scope
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('model.h5')

# Define parameters
img_size = (224, 224)

# Define class labels (you can adjust this based on your dataset)
class_labels = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70', 'speed limit 80', 'speed limit 100', 'speed limit 120', 'stop sign',]

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    return img_array


    @app.route('/')
    def index():
        return render_template('index.html')

# Define a route to handle image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})

    # Save the image to a temporary location
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    # Preprocess the image
    img_array = preprocess_image(temp_image_path)

    # Perform classification
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    return jsonify({'class': predicted_class})

# Define a main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
