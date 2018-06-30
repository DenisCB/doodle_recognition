import os
from flask import Flask, render_template, request
from model.predictor import Predictor
# from flask_cors import CORS, cross_origin
# CORS(app, headers=['Content-Type'])
# app.config["CACHE_TYPE"] = "null"

app = Flask(__name__)
predictor = Predictor(path='model/')


@app.route('/', methods=["POST", "GET", "OPTIONS"])
def main_page():
    return render_template('index.html')


@app.route('/prediction_page', methods=["POST"])
def predict_img():
    predictor.decode_image(request)
    predictor.process_image()
    result = predictor.predict_image()
    return result


@app.route('/get_ideas', methods=["POST"])
def get_some_ideas():
    lines, line = [], ''
    for word in predictor.get_classes_example(num_classes=5):
        if len(line) + len(word) >= 35:
            lines.append(line)
            line = ''
        line += word + ', '
    lines.append(line[:-2])
    return '<br>'.join(lines)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
