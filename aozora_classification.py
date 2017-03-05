# coding: utf-8

"""
This predict how much a sentence is similar to one of the Japanese classical author.
Before using this, be sure to generate a model using aozora_cnn.py.
Change the values for model_file to your model hdf5 file path, raw_txt to your favorite sentence in Japanese
and target_author to the author you trained the model.
"""

import numpy as np
import pandas as pds
from keras.models import load_model

model_file = "/tmp/weights.07-0.22-0.92-0.35-0.88.hdf5"
raw_txt = "隴西の李徴は博學才穎、天寶の末年、若くして名を虎榜に連ね、ついで江南尉に補せられたが、性、狷介、自ら恃む所頗る厚く、賤吏に甘んずるを潔しとしなかつた。"
target_author = ["夏目漱石","芥川龍之介","森鴎外","坂口安吾"]



# Encodes the raw_txt
def text_encoding(raw_txt):
    txt = [ord(x) for x in str(raw_txt).strip()]
    txt = txt[:200]
    if len(txt) < 200:
        txt += ([0] * (200 - len(txt)))
    return txt



# Predict
def predict(comments, model_filepath="model.h5"):
    model = load_model(model_filepath)
    ret = model.predict(comments)
    return ret




if __name__ == "__main__":
    txt = text_encoding(raw_txt)
    predict_result = predict(np.array([txt]), model_filepath=model_file)

    pds_predict_result = pds.DataFrame(predict_result, columns=target_author)




pds_predict_result


