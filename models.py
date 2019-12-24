from app import db
import datetime


def generate_filename():
    max_id = Drawings.query.order_by(-Drawings.id).first().id + 1
    return 'doodles/{}_{}.jpg'.format(max_id, datetime.datetime.now())


class Drawings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    predicted_label = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now())
    s3_filename = db.Column(db.String(100), nullable=True)

    def __init__(self, *args, **kwargs):
        super(Drawings, self).__init__(*args, **kwargs)
        self.s3_filename = generate_filename()

    def __repr__(self):
        return "<Drawing {}, label: {}, confidence: {}, file: {}>".format(
            self.id, self.predicted_label, round(self.confidence, 3),
            self.s3_filename
        )
