import numpy as np
import os
from sklearn.cross_validation import train_test_split
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import csv
from itertools import izip


from skimage import data
import glob

from keras import backend

#backend.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

if __name__ == '__main__':
    data_path = "C:/Users/Pedro/Desktop/Project/Data"  # This folder holds the csv files

    # load csv files. We use np.loadtxt. Delimiter is ","
    # and the text-only header row will be skipped.

    print("Loading data...")
    x_train = np.loadtxt(data_path + os.sep + "x_train.csv",
                         delimiter=",", skiprows=1)
    x_test = np.loadtxt(data_path + os.sep + "x_test.csv",
                        delimiter=",", skiprows=1)
    y_train = np.loadtxt(data_path + os.sep + "y_train.csv",
                         delimiter=",", skiprows=1)

    print "All files loaded. Preprocessing..."

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

    # Reshape by raveling each 100x5 array into a 500-length vector
    x_train = [g.ravel() for g in x_train]
    x_test = [g.ravel() for g in x_test]

    # convert data from list to array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_train = np.ravel(y_train)

    # Now x_train should be 15485 x 500 and x_test 3871 x 500.
    # y_train is 15485-long vector.

    print("x_train shape is %s" % str(x_train.shape))
    print("y_train shape is %s" % str(y_train.shape))
    print("x_test shape is %s" % str(x_test.shape))

    print('Data preprocessing done...')

    # print("Next steps FOR YOU:")
    print("-" * 30)

    # print("1. Define a classifier using sklearn")

    x_train_, x_test_, y_train_, y_test_ = train_test_split(x_train, y_train, test_size=0.2)

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

    # clf_list = [LogisticRegression(), SVC()]
    # clf_name = ['LR', 'SVC']
    # C_range = 10.0 ** np.arange(-5, 0)
    # for clf, name in zip(clf_list, clf_name):
    #     for C in C_range:
    #         for penalty in ["l1", "l2"]:
    #             clf.C = C
    #             clf.penalty = penalty
    #             clf.fit(x_train_, y_train_)
    #             y_pred = clf.predict(x_test)
    #             score = accuracy_score(y_test_, y_pred)
    #             print "Score/C/Penalty", score, C, penalty

    # **************

    # clf = LogisticRegression()
    # clf.C = 0.00001
    # penalty = "l2"
    # clf.penalty = penalty
    # clf.fit(x_train_, y_train_)
    # y_pred = clf.predict(x_test_)
    # score = accuracy_score(y_test_, y_pred)
    # print score


    # ***********

#           NEURAL NETWORK

    N = 50  # Number of feature maps
    width = 10, # Conv. window size
    data_shape = (100,5)
    model = Sequential()

    model.add(
        Convolution1D(nb_filter=N, filter_length= width, border_mode='same', input_shape= data_shape , activation='relu'))

    model.add(MaxPooling1D(5))

    model.add(Convolution1D(nb_filter=N, filter_length= width, border_mode='same', activation='relu'))

    model.add(MaxPooling1D(5))

    model.add(Flatten())

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train_, y_train_, nb_epoch=5, batch_size=16)
    y_test_2 = model.predict(x_test_, batch_size=32, verbose=0)
    print accuracy_score(y_test_,y_test_2)

#            END OF NEURAL NETWORK


    # print("2. Assess its accuracy using cross-validation (optional)")
    # print("3. Fine tune the parameters and return to 2 until happy (optional)")
    # print("4. Create submission file. Should be similar to y_train.csv.")

    # with open(data_path + os.sep + "y_test.csv", 'wb') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerows(izip(['GeneId'],['Prediction']))
    #     y_test = [int(i) for i in y_test]
    #     writer.writerows(izip(range(1, len(y_test)+1), y_test))
    # print("5. Submit at kaggle.com and sit back.")