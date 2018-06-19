from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import base64
import numpy as np
from PIL import Image
import keras


all_classes = np.load('ML/classes.npy')
model =  keras.models.load_model('ML/nnet_v1_recognized.h5')
model._make_predict_function()

app = Flask(__name__)
CORS(app, headers=['Content-Type'])
# app.config["CACHE_TYPE"] = "null"


@app.route('/', methods=["POST", "GET", "OPTIONS"])
def main_page():
    return render_template('index.html')


@app.route('/hook', methods=["GET", "POST", 'OPTIONS'])
def get_image():
    if request.method == 'POST':
        image_b64 = request.values['imageBase64']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))
        with open('tmp/f1.jpg', 'wb') as f:
            f.write(image)
    return 'Picture saved'

@app.route('/predict', methods=["GET", "POST", 'OPTIONS'])
def predict_img():
    if request.method != 'POST':
        return 'Something is wrong'

    image_b64 = request.values['imageBase64']
    image_encoded = image_b64.split(',')[1]
    image = base64.decodebytes(image_encoded.encode('utf-8'))
    with open('tmp/img.jpg', 'wb') as f:
        f.write(image)
    image = Image.open('tmp/img.jpg')


    results = []
    for resample in [Image.HAMMING, 3]:
        bbox = Image.eval(image, lambda px: 255-px).getbbox()
        img = image.crop(bbox).resize((56, 56), resample=resample)
        img = np.array(img)[:, :, 0]
        img = - img + img.max()
        img = img.clip(0, int(img.max()/2))
        preds = model.predict(img.reshape(1, 56, 56, 1))

        if np.isnan(img).any():
            results.append('Drew something first.')
        elif preds.max() < 0.5:
            results.append('I have no idea what you drew.')
        else:
            results.append("I'm {}% sure it's a {}.".format(
                round(100*preds.max(), 1),  all_classes[preds.argmax()]))

    return '<br>'.join(results)


@app.route('/get_ideas', methods=["GET", "POST", 'OPTIONS'])
def get_some_ideas():
    #result = '<br>'.join(np.random.choice(all_classes, 5, False)) + '<br>'

    lines = []
    line = ''
    for w in sorted(np.random.choice(all_classes, 10, True)):
        if len(line) + len(w) >= 38:
            lines.append(line)
            line = ''
        line += w + ', '
    lines.append(line[:-2])

    return '<br>'.join(lines) + '.<br>'


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


if __name__ == "__main__":
    app.config["CACHE_TYPE"] = "null"

    app.run()
