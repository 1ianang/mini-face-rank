# coding: utf8
from scipy.stats import pearsonr
import pickle
import random
import numpy as np
from keras import Sequential, Model
from keras.layers import Dense
from keras.models import load_model

def load_data():
    with open("training.pkl", "rb") as fi:
        return pickle.load(fi)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def error_stat():
    faces = load_data()
    total_faces = len(faces)
    pos = int(total_faces * 0.8)
    faces = faces[pos:]
    x = np.array([item['enc'] for item in faces])
    y = np.array([int(item['score']) for item in faces])
    model = load_model("aq.h5")
    predicted = model.predict(x)
    print("predicted shape: ", predicted.shape)
    predicted = np.squeeze(predicted)
    print("predicted shape: ", predicted.shape)
    predicted = np.array([int(score) for score in predicted])
    errors = np.where(y != predicted)[0]
    print(errors.shape)
    print("Errors: %d/%d" % (errors.shape[0], x.shape[0]))

def pearson_corr():
    faces = load_data()
    total_faces = len(faces)
    pos = int(total_faces * 0.8)
    faces = faces[pos:]
    x = np.array([item['enc'] for item in faces])
    y = np.array([item['score'] for item in faces])
    model = load_model("face_rank_model.h5")
    predicted = model.predict(x)
    print("predicted shape: ", predicted.shape)
    predicted = np.squeeze(predicted)
    print("predicted shape: ", predicted.shape)
    predicted = np.array(predicted)
    corr, p = pearsonr(y, predicted)
    print("corr: %.4f" % corr)
    print("rmse: %.4f" % rmse(y, predicted))

if __name__ == '__main__':
    pearson_corr()
    #error_stat()


