from app import app
from flask import render_template, request
from model.predictor import Predictor

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
        if len(line) + len(word) >= 40:
            lines.append(line)
            line = ''
        line += word + ', '
    lines.append(line[:-2])
    return '<br>'.join(lines)
