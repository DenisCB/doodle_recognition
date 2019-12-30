import numpy as np
from app import app, db
from models import Drawings
from flask import render_template, request, jsonify
from model.predictor import Predictor
from helpers.bucket_helper import BucketHelper


predictor = Predictor(path='model/')
bucket_helper = BucketHelper()


@app.route('/', methods=["POST", "GET", "OPTIONS"])
def main_page():
    return render_template('index.html')


@app.route('/prediction_page', methods=["POST"])
def predict_img():
    predictor.decode_image(request)
    predictor.process_image()
    predicted_label, confidence, message = predictor.predict_image()

    response = {
        'message': message,
        'predicted_label': predicted_label,
        'confidence': float(confidence),
    }
    return jsonify(response)


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


@app.route('/save_img', methods=["POST"])
def save_img():
    drawing = Drawings(
        predicted_label=request.values['predicted_label'],
        confidence=float(request.values['confidence'])
    )
    db.session.add(drawing)
    db.session.commit()

    predictor.decode_image(request)
    bucket_helper.upload_img(predictor.src_image, drawing.s3_filename)

    return 'Saved'


@app.route('/drawings', methods=["GET"])
def what_others_drew():
    num_drawings = 4
    max_id = Drawings.max_id()
    ids_selected, drawings = [], []
    while len(drawings) < num_drawings:
        id = np.random.randint(49, max_id+1)
        if id in ids_selected:
            continue
        drawing = Drawings.query.get(id)
        if drawing and drawing.s3_filename and 'None' not in drawing.s3_filename:
            drawings.append(drawing)
            ids_selected.append(id)

    return render_template('drawings.html', drawings=drawings)
