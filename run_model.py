from tensorflow.keras import layers, metrics, losses, optimizers, callbacks, regularizers
from tensorflow import keras
import tensorflow as tf
import os
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

suffix = ''

def get_model():
    # 定义卷积层

    # 核大小
    filters = 4
    kernel_size = 4

    convolution_1d_layer = keras.layers.Conv1D(filters, kernel_size,
                                               input_shape=(500, 1),
                                               strides=1, padding='same',
                                               activation="relu",
                                               name="convolution_1d_layer")

    # 最大池化层
    max_pooling_layer = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name="max_pooling_layer")

    # 展平层
    reshape_layer = keras.layers.Flatten(name="reshape_layer")

    # dropout 层
    dropout_layer = keras.layers.Dropout(0.5, name="dropout_layer")

    # 全连接层
    full_connect_layer = keras.layers.Dense(128, activation="relu", name="full_connect_layer")

    model = keras.Sequential()
    model.add(convolution_1d_layer)
    model.add(max_pooling_layer)
    model.add(reshape_layer)
    model.add(dropout_layer)
    model.add(full_connect_layer)
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss='binary_crossentropy',
        metrics=['accuracy', metrics.AUC()]
    )
    return model

def train_fold(model: keras.Model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=35,
        verbose=1,
        validation_data=(x_val, y_val),
        shuffle=True
    )
    pred = model.predict(x=x_test, batch_size=32)
    return pred

if __name__ == "__main__":

    x = np.loadtxt('../feature1/data.txt')
    y = np.loadtxt('../feature1/label.txt')
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, 1)

    kfold = StratifiedKFold(n_splits=10, random_state=222, shuffle=True)
    test_auc, test_pr, preds = list(), list(), np.zeros((x.shape[0],))
    for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
        print("-" * 10 + "No. :", fold + 1)

        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = x[train_index], x[test_index]

        model = get_model()
        if fold == 0:
            model.summary()

        pred = train_fold(model, x_train, y_train, x_test, y_test, x_test, y_test)
        preds[test_index] = pred[:, 0]
        test_auc.append(roc_auc_score(y_test, pred))
        precision, recall, thresholds = precision_recall_curve(y_test, pred)
        test_pr.append(auc(recall, precision))
        print("-" * 10 + "No. :", fold + 1)
        print("auroc = {:.4f} ".format(roc_auc_score(y_test, pred)) + "aupr = {:.4f} ".format(auc(recall, precision)))

        keras.backend.clear_session()
    print("10 fold auc :", test_auc)
    print("10 fold mean auroc {:.5f}".format(np.mean(test_auc)))
    print("10 fold mean aupr {:.5f}".format(np.mean(test_pr)))
