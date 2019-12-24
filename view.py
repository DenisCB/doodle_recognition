from app import app
from flask import render_template, request
from model.predictor import Predictor
from models import Drawing
from app import db


predictor = Predictor(path='model/')


@app.route('/', methods=["POST", "GET", "OPTIONS"])
def main_page():
    return render_template('index.html')


@app.route('/prediction_page', methods=["POST"])
def predict_img():
    predictor.decode_image(request)
    predictor.process_image()
    predicted_label, confidence, message = predictor.predict_image()

    drawing = Drawing(
        predicted_label=predicted_label,
        confidence=confidence
    )
    db.session.add(drawing)
    db.session.commit()

    return message


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
