from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2
import os

def load_image(path):
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224,224)).astype(np.float32)
    return x

def read_data(data_path, valid_size=0.0):
    path_images = []
    labels = []

    for path in os.listdir(data_path):
        for img_path in os.listdir(os.path.join(data_path, path)):
            path_images.append(os.path.join(data_path, path, img_path))
            labels.append(path)
    
    images = np.array([load_image(impath) for impath in path_images])
    labels = np.array(labels)

    if valid_size != 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, 
            test_size=valid_size, 
            random_state=7, 
            stratify=labels)

        return X_train, X_val, y_train, y_val

    return images, labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, '{}'.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return None

def standard_scaler(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))