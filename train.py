# coding: utf8
import pickle
import random
import numpy as np
from keras import Sequential, Model
from keras.layers import Dense

def load_data():
    with open("training.pkl", "rb") as fi:
        return pickle.load(fi)

def split_data(faces_with_label, train_percent = 0.8):
    pos = int(len(faces_with_label) * train_percent)
    print("Split pos: ", pos)
    x = [item['enc'] for item in faces_with_label]
    print("enc size: ", x[0].shape)
    y = [item['score'] for item in faces_with_label]
    x = np.array(x)
    y = np.array(y)
    print("shape of x: ", x.shape)
    print("shape of y: ", y.shape)
    train_x, train_y = x[:pos], y[:pos]
    val_x, val_y = x[:pos], y[:pos]
    return (train_x, train_y), (val_x, val_y)

def build_model(input_dim, hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=input_dim, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def main():
    train_batch_size = 64
    epochs = 7500
    faces_with_score = load_data()
    #random.shuffle(faces_with_score)
    print("Total faces: %d" % len(faces_with_score))
    (train_x, train_y), (val_x, val_y) = split_data(faces_with_score)
    model = build_model(input_dim=128, hidden_size=512)
    history = model.fit(train_x, train_y, batch_size=train_batch_size, epochs=epochs, shuffle=False, validation_data=(val_x, val_y))
    model.evaluate(val_x, val_y)
    model.save("face_rank_model.h5")


if __name__ == '__main__':
    main()


