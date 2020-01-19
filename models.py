from app import db
import datetime


def generate_filename():
    last_obj = Drawings.query.order_by(-Drawings.id).first()
    if last_obj:
        max_id = last_obj.id + 1
    else:
        max_id = 1
    return 'doodles/{}_{}.jpg'.format(max_id, datetime.datetime.now())


class Drawings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    actual_label = db.Column(db.String(100), nullable=True)
    predicted_label = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=db.func.now())
    s3_filename = db.Column(db.String(100), nullable=True)

    def __init__(self, *args, **kwargs):
        super(Drawings, self).__init__(*args, **kwargs)
        self.s3_filename = generate_filename()

    def __repr__(self):
        return "<Drawing {}, label: {}, confidence: {}, file: {}>".format(
            self.id, self.predicted_label, round(self.confidence, 3),
            self.s3_filename
        )

    @classmethod
    def max_id(self):
        return db.session.query(db.func.max(Drawings.id)).first()[0]
