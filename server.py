import numpy as np
from flask import Flask
from werkzeug.utils import secure_filename
from fastai import *
from flask import request
from fastai.vision.all import *
from fastai.imports import *
from fastai.vision import *
from io import BytesIO
from flask import jsonify

app = Flask(__name__)
      
def setup_learner():
    #await download_file(export_file_url, path / export_file_name)
    path = "models/model_low_resnet50.pkl"
    try:
        learn = load_learner(path)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

learn = setup_learner()


@app.route("/", methods=["GET"])
def index():
    return jsonify({'result': "hola"})


@app.route('/analyze', methods=['POST'])
def analyze():
    img_data = request.files['file']
    img_file = secure_filename(img_data.filename)
    img_data.save(img_file)
    img_bytes = img_data.read()
    img = PILImage.create(img_data)
    prediction = learn.predict(img)[0]

    return jsonify({'result': str(prediction)})


if __name__ == '__main__':
    #if 'serve' in sys.argv:
    app.run(port=5000)