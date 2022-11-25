import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

print('Model loaded. Check http://127.0.0.1:7403/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds

def connect_to_database(host, port, user, password, database):
    import sqlalchemy as db
    import sys

    # specify connection string
    connection_str = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    # connect to database
    engine = db.create_engine(connection_str)
    try:
        connection = engine.connect()
        print('CONNECTED TO DATABASE!', file=sys.stderr)
        print('', file=sys.stderr)
        return True
    except Exception as e:
        print('UNABLE TO CONNECT TO DATABASE!', file=sys.stderr)
        print('Exception:', file=sys.stderr)
        print(e, file=sys.stderr)
        print('', file=sys.stderr)
        return False

@app.route('/', methods=['GET'])
def index():
    # Main page
    connected = connect_to_database(host='imgprocessing', port=3306, user='researcher',
                                    password='science', database='images')
    if connected:
        return render_template("index_connected.html")
    else:
        return render_template("index_not_connected.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    host = '0.0.0.0'
    port = 7403
    http_server = WSGIServer((host, port), app)
    http_server.serve_forever()
