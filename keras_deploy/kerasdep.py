import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:7403/')

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
        return jsonify(result='Goldfish', probability='99.999')

    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7403)
