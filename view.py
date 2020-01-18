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


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


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
    print(request.values['predicted_label'])
    print(request.values['confidence'])
    print(request.values['actual_label'] or request.values['predicted_label'])

    drawing = Drawings(
        actual_label=(
            request.values['actual_label']
            or request.values['predicted_label']
        ),
        predicted_label=request.values['predicted_label'],
        confidence=float(request.values['confidence'])
    )
    db.session.add(drawing)
    db.session.commit()

    predictor.decode_image(request)
    bucket_helper.upload_img(predictor.src_image, drawing.s3_filename)

    return ''


@app.route('/drawings/<kind>', methods=["GET"])
def what_others_drew(kind):
    # kind is either recent or random
    num_drawings = 4
    max_id = Drawings.max_id()
    drawings = {}

    def add_if_valid(drawings, id):
        if id in drawings:
            return False
        drawing = Drawings.query.get(id)
        if (
            drawing
            and drawing.s3_filename
            and 'None' not in drawing.s3_filename
        ):
            drawings[id] = drawing
        return drawings

    if kind == 'random':
        while len(drawings) < num_drawings:
            id = np.random.randint(49, max_id+1)
            drawings = add_if_valid(drawings, id)
    elif kind == 'recent':
        id = max_id
        while len(drawings) < num_drawings:
            drawings = add_if_valid(drawings, id)
            id -= 1
    return render_template('drawings.html', drawings=drawings.values())
