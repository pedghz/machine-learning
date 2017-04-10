import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import csv


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if __name__ == '__main__':
    data_path = "./train"  # This folder holds the csv files

    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.

    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv",
                         delimiter=",", skiprows=1)
    x_test = np.loadtxt(data_path + os.sep + "x_test.csv",
                        delimiter=",", skiprows=1)
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv",
                         delimiter=",", skiprows=1)

    print("All files loaded. Preprocessing...")

    # remove the first column(Id)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]
    y_train = y_train[:, 1:]

    # Every 100 rows correspond to one gene.
    # Extract all 100-row-blocks into a list using np.split.
    num_genes_train = x_train.shape[0] / 100
    num_genes_test = x_test.shape[0] / 100

    print("Train / test data has %d / %d genes." % \
          (num_genes_train, num_genes_test))
    x_train = np.split(x_train, num_genes_train)
    x_test = np.split(x_test, num_genes_test)

    #    # Reshape by raveling each 100x5 array into a 500-length vector
    #    x_train = [g.ravel() for g in x_train]
    #    x_test = [g.ravel() for g in x_test]

    # convert data from list to array
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    X_test = np.array(x_test)
    y_train = np.ravel(y_train)

    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.

    #    print("x_train shape is %s" % str(x_train.shape))
    #    print("y_train shape is %s" % str(y_train.shape))
    #    print("x_test shape is %s" % str(x_test.shape))

    print('Data preprocessing done...')

    # print("Next steps FOR YOU:")
    print("-" * 30)

    # X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, test_size=0.2)

    # print("1. Define a classifier using sklearn")

    # classifiers = [(RandomForestClassifier(), "Random Forest"),
    #                (ExtraTreesClassifier(), "Extra-Trees"),
    #                (AdaBoostClassifier(), "AdaBoost"),
    #                (GradientBoostingClassifier(), "GB-Trees")]
    # accuracies = []
    # for clf, name in classifiers:
    #     clf.n_estimators = 100
    #     clf.fit(x_train_, y_train_)
    #     y_hat = clf.predict(x_test_)
    #     accuracy = accuracy_score(y_test_, y_hat)
    #     accuracies.append(accuracy)

    # **************

    clf_list = [LogisticRegression(), SVC()]
    clf_name = ['LR', 'SVC']
    C_range = 10.0 ** np.arange(-5, 0)
    for clf, name in zip(clf_list, clf_name):
        for C in C_range:
            for penalty in ["l1", "l2"]:
                clf.C = C
                clf.penalty = penalty
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

                skf = StratifiedKFold(y_train, 15, shuffle=True)
                scores = cross_val_score(clf, x_train, y_train)
                print("C = %.2e and penalty = %s - Accuaracy is: %.2f %% +- %.2f" % (C, penalty, 100 * np.mean(scores), 100 * np.std(scores)))


    #              NEURAL NETWORK

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 2)
    N = 50
    width = 10
    data_shape = (100, 5)
    model = Sequential()
    model.add(Convolution1D(N, width, border_mode='valid', input_shape=data_shape, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Convolution1D(nb_filter=N, filter_length=width, border_mode='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, nb_epoch=13, batch_size=32)
    probas = model.predict_proba(X_test, verbose=1)

    #            END OF NEURAL NETWORK


    #            Making CSV file

    with open("cnn_purified_test.csv", "w") as output_handler:
        output_handler.write("GeneId,Prediction\n")
        for i in range(len(probas)):
            output_handler.write(str(i + 1) + "," + str(probas[i, 1]) + '\n')

    # **************
