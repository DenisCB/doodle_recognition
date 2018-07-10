import numpy as np
import PIL
from PIL import Image, ImageDraw

px, border_px = np.load('../model/processing_params.npy')


def im2arr(drawing):
    scale = np.random.beta(6, 2)

    # Original images are 255x255, add extra 5 to each edge.
    im = PIL.Image.new(mode='L', size=(260, 260))
    draw = ImageDraw.Draw(im)

    # Shift the strokes from edges by 5 pixels, convert them to valid format.
    for stroke in drawing:
        stroke_shifted = list(map(lambda x: tuple([(i+2.5)*scale for i in x]),
                                  tuple(zip(stroke[0], stroke[1]))))
        draw.line(stroke_shifted, fill=255, width=4)

    # Find the bounding box.
    bbox = PIL.Image.eval(im, lambda x: x).getbbox()
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
        bbox[0]-border_px, bbox[1]-border_px,
        bbox[2]+border_px, bbox[3]+border_px)

    # Crop and resize.
    im = im.crop(bbox)
    im = im.resize((px, px), resample=3)

    # Clip max values to make lines less blury.
    im = np.array(im).astype(float)
    im /= im.max()/2

    return im.clip(0, 1)
