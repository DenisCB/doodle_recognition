from app import db
import datetime


class Drawing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    predicted_label = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
