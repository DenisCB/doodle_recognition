import numpy as np
from PIL import Image, ImageOps
import keras
import base64
from ML.torch_models import InceptionBasedNet
import torch


class Predictor(object):

    def __init__(self, path):
        self.all_classes = np.load(path+'classes.npy')
        self.px, self.border_px = np.load(path+'processing_params.npy')
        self.mean_global = np.load(path+'mean_global.npy')[0]
        self.model = keras.models.load_model(path+'nnet_96_aug_v1.h5')
        self.model._make_predict_function()

        self.torch_model = InceptionBasedNet(
            self.all_classes.shape[0], (1, self.px, self.px))
        self.torch_model.load_state_dict(
            torch.load(
                path+'InceptionBasedNet.t7',
                map_location='cpu'))
        self.torch_model.eval()
        self.image = None

    def decode_image(self, request):
        image = request.values['image'].split(',')[1]
        image = base64.decodebytes(image.encode('utf-8'))
        filename = 'tmp/img_1.jpg'
        with open(filename, 'wb') as f:
            f.write(image)
        image = Image.open(filename)
        self.image = image

    def process_image(self):
        # Invert colors, since in PIL white is 255 and black is 0.
        img = self.image.convert('L')
        img = ImageOps.invert(img)

        # Find the bounding box.
        bbox = Image.eval(img, lambda x: x).getbbox()
        if bbox is None:
            self.image = None
            return

        width = bbox[2] - bbox[0]  # right minus left
        height = bbox[3] - bbox[1]  # bottom minus top
        # Center after croping.
        diff = width - height
        if diff >= 0:
            bbox = (bbox[0], bbox[1]-diff/2, bbox[2], bbox[3]+diff/2)
        else:
            bbox = (bbox[0]+diff/2, bbox[1], bbox[2]-diff/2, bbox[3])
        # Add borders.
        bbox = (
            bbox[0]-self.border_px, bbox[1]-self.border_px,
            bbox[2]+self.border_px, bbox[3]+self.border_px)

        # Crop and resize.
        img = img.crop(bbox)
        img = img.resize((self.px, self.px), resample=3)
        img = np.array(img).astype(float)

        # Clip max values to make lines less blury.
        img /= img.max()/2
        self.image = img.clip(0, 1) - self.mean_global

    def predict_image(self):
        if self.image is None:
            return 'Draw something first.'

        result = []
        for mode in ['Keras', 'Torch']:
            if mode == 'Keras':
                nnet_name = 'Keras, convolution nnet'
                img = self.image.reshape(1, self.px, self.px, 1)
                preds = self.model.predict(img)[0]
            elif mode == 'Torch':
                nnet_name = 'Torch, Inception v3 nnet'
                img = self.image.reshape(1, 1, self.px, self.px)
                img = torch.Tensor(img)
                preds = self.torch_model(img).detach().numpy()

            res = nnet_name + ':<br>'
            if preds.max() < 0.1:
                res += "I have no idea what you drew."
            else:
                res += "I'm {}% sure it's a {}.".format(
                    round(100*preds.max(), 1), self.all_classes[preds.argmax()]
                    )
            result.append(res+'<br>')
        return "<br>".join(result)

    def get_classes_example(self, num_classes):
        return sorted(np.random.choice(self.all_classes, num_classes, False))
