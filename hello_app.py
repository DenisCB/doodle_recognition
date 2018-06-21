from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import base64
import numpy as np
from PIL import Image, ImageOps
import keras


all_classes = np.load('ML/classes.npy')
model =  keras.models.load_model('ML/nnet_v1.h5')
model._make_predict_function()
border = 2
px = 64

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
        # Invert colors.
        img = image.convert('L')
        img = ImageOps.invert(img)

        # Find the bounding box.
        bbox = Image.eval(img, lambda x: x).getbbox()
        if bbox is None:
            return '<br>Drew something first.'
        width = bbox[2] - bbox[0] # right minus left
        height = bbox[3] - bbox[1] # bottom minus top
        # Center after croping.
        diff = width - height
        if diff >= 0:
            bbox = (bbox[0], bbox[1]-diff/2, bbox[2], bbox[3]+diff/2)
        else:
            bbox = (bbox[0]+diff/2, bbox[1], bbox[2]-diff/2, bbox[3])
        # Add borders.
        bbox = (bbox[0]-border, bbox[1]-border, bbox[2]+border, bbox[3]+border)

        # Crop and resize.
        img = img.crop(bbox)
        img = img.resize((px, px), resample=resample)
        img = np.array(img)

        # Clip max values to make lines less blury.
        img = img.clip(0, img.max()/2)


        preds = model.predict(img.reshape(1, px, px, 1))
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
