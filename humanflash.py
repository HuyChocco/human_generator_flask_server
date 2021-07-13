
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import base64
import io
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import time

import tensorflow as tf
from PIL import Image
from flask_restful import Resource, Api, reqparse
from tensorflow.keras.preprocessing.image import save_img

app = Flask(__name__)
api = Api(app)


model = load_model('fully_trained.h5')

print('*  MODEL LOADED')


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert('RBG')

    image.resize(target)
    image = img_to_array(image)

    image = (image-127.5)/127.5
    image = np.expand_dims(image, axis=0)
    return image


class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        img_data = json_data['image']

        image = base64.b64decode(str(img_data))

        img = Image.open(io.BytesIO(image))

        prepared_image = prepare_image(img, target=(256, 256))

        preds = model.predict(prepared_image)

        outputfile = 'output.png'
        save_path = './output/'

        output = tf.reshape(preds, [256, 256, 3])

        ouput = (output+1)/2

        save_img(save_path+outputfile, img_to_array(ouput))

        imageNew = Image.open(save_path+outputfile)
        imageNew = imageNew.resize((50, 50))
        imageNew.save(save_path+'new_'+outputfile)

        with open(save_path+'new_'+outputfile, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())

        ouputData = {
            'Image': str(encoded_string)
        }

        return ouputData


api.add_resource(Predict, '/predict')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
