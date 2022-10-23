from fastapi import File, UploadFile, Form, HTTPException
from fastapi import FastAPI
import boto3
import pickle
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import models

app = FastAPI()

s3 = boto3.resource('s3')


def img_preprocessing(img):
    print(img)
    image = keras.preprocessing.image.load_img(img)
    image_arr = keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr, (224, 224)).numpy()
    image_arr /= 255

    return image_arr


def tf_preprocess(img):
    model = load_model()
    train_input_vvg16 = preprocess_input(np.array([img]))
    train_features = model.predict(train_input_vvg16)
    return train_features


def load_model():
    base_model = VGG16(weights='imagenet', include_top=True)
    x = base_model.get_layer('fc2').output
    model = models.Model(inputs=base_model.input, outputs=x)
    return model


@app.get("/")
def read_root():
    return {"Hackathon": "BBVA"}


@app.post("/login")
def read_item(hand_left_palmer: Union[UploadFile, None] = None,
              hand_right_palmer: Union[UploadFile, None] = None,
              hand_left_dorsal: Union[UploadFile, None] = None,
              hand_right_dorsal: Union[UploadFile, None] = None):
    features_list = []
    features_type = []

    if hand_left_palmer:
        out_file = open(hand_left_palmer.filename, 'wb')
        out_file.write(hand_left_palmer.file.read())
        out_file.close()

        features_type.append(2)
        features_list.append(img_preprocessing(hand_left_palmer.filename))

    if hand_right_palmer:
        out_file = open(hand_right_palmer.filename, 'wb')
        out_file.write(hand_right_palmer.file.read())
        out_file.close()

        features_type.append(3)
        features_list.append(img_preprocessing(hand_right_palmer.filename))

    if hand_left_dorsal:
        out_file = open(hand_left_dorsal.filename, 'wb')
        out_file.write(hand_left_dorsal.file.read())
        out_file.close()

        features_type.append(0)
        features_list.append(img_preprocessing(hand_left_dorsal.filename))

    if hand_right_dorsal:
        out_file = open(hand_right_dorsal.filename, 'wb')
        out_file.write(hand_right_dorsal.file.read())
        out_file.close()

        features_type.append(1)
        features_list.append(img_preprocessing(hand_right_dorsal.filename))

        # s3.Bucket('bbva-hackathon').put_object(Key=hand.filename, Body=hand.file)

    if len(features_list) > 0:
        clf = pickle.load(open('model_classification.pkl', 'rb'))

        features = tf_preprocess(np.array(features_list))

        features = pd.DataFrame(features)
        types_img = pd.DataFrame(features_type)

        features = pd.concat([features, types_img], axis=1)

        features = clf.predict(features)
        features = pd.DataFrame(features)
        return json.dump({"IDS": features})
    else:
        raise HTTPException(status_code=400, detail='At Least one photo is necessary!')


def read_item(hand: UploadFile, type_hand: str = Form()):
    if hand:
        out_file = open(hand.filename, 'wb')
        out_file.write(hand.file.read())
        out_file.close()

        s3.Bucket('bbva-hackathon').put_object(Key=hand.filename, Body=hand.file)

        clf = pickle.load(open('model_classification.pkl', 'rb'))

        features = img_preprocessing(hand.filename)
        features = tf_preprocess(features)
        features = pd.DataFrame(features)

        type_df = pd.DataFrame([int(type_hand)])
        features = pd.concat([features, type_df], axis=1)

        pred = clf.predict(features)

        return {"ID": str(pred[0])}

    return {'Error': 'Hand File is required'}


@app.post("/updateModel")
def update_model():
    s3_client = boto3.client('s3')
    s3_client.download_file('bbva-hackathon', 'model_classification.pkl', 'model_classification.pkl')

    return {}
