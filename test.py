from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn import metrics

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle

import utils

def test(model, X_test, y_test, enc):

    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred,axis=1)

    y_test_pred_labels = enc.inverse_transform(y_test_pred)

    cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred_labels, labels=enc.classes_)
    plt.figure()
    utils.plot_confusion_matrix(cm, enc.classes_, normalize=False)
    plt.show()

    test_acc = metrics.accuracy_score(enc.transform(y_test), y_test_pred)
    print("Accuracy: {}".format(test_acc))

if __name__ == "__main__":

    X_test, y_test = utils.read_data("data/data3a/validation/")
    X_test = np.array(list(map(preprocess_input, X_test)))

    enc = pickle.load(open("models/label_enc.pkl", "rb"))
    model = tf.keras.models.load_model("models/model_92acc.h5")

    test(model, X_test, y_test, enc)