import numpy as np
from PIL import Image, ImageOps
import keras
import os

from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import base64


all_classes = np.load('ML/classes.npy')
model =  keras.models.load_model('ML/nnet_96_aug_v1.h5')
model._make_predict_function()
border = 2
px = 96
global_mean = 0.09752

app = Flask(__name__)
CORS(app, headers=['Content-Type'])
#app.config["CACHE_TYPE"] = "null"


@app.route('/', methods=["POST", "GET", "OPTIONS"])
def main_page():
    return render_template('index.html')


@app.route('/prediction_page', methods=["POST"])
def predict_img():

    # Decode, save and load the picture.
    image_b64 = request.values['imageBase64']
    image_encoded = image_b64.split(',')[1]
    image = base64.decodebytes(image_encoded.encode('utf-8'))

    # files_numbers = [
    #     int(f[4:-4]) for f in os.listdir('tmp/') 
    #     if f.endswith('.jpg')]
    # next_filename = 'tmp/img_'+str(max(files_numbers)+1)+'.jpg'
    next_filename = 'tmp/img_1.jpg'
    with open(next_filename, 'wb') as f:
        f.write(image)
    image = Image.open(next_filename)


    results = []
    for resample in [Image.HAMMING]:
        # Invert colors, since in PIL white is 255 and black is 0.
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
        img = np.array(img).astype(float)

        # Clip max values to make lines less blury.
        img /= img.max()/2
        igm = img.clip(0, 1) - global_mean

        preds = model.predict(img.reshape(1, px, px, 1))
        if np.isnan(img).any():
            results.append('Drew something first.')
        elif preds.max() < 0.1:
            results.append('I have no idea what you drew.')
        else:
            results.append("I'm {}% sure it's a {}.".format(
                round(100*preds.max(), 1),  all_classes[preds.argmax()]))

    return '<br>'.join(results)


@app.route('/get_ideas', methods=["GET", "POST", 'OPTIONS'])
def get_some_ideas():

    lines = []
    line = ''
    for w in sorted(np.random.choice(all_classes, 10, False)):
        if len(line) + len(w) >= 38:
            lines.append(line)
            line = ''
        line += w + ', '
    lines.append(line[:-2])

    return '<br>'.join(lines) + '.<br>'


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)