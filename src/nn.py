from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import shutil

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, ShuffleSplit

import deepxde as dde
from data import BerkovichData, ExpData, FEMData, ModelData, FEMDataT, BerkovichDataT, ExpDataT

import os

def svm(data):

    clf = SVR(kernel="rbf")
    clf.fit(data.train_x, data.train_y[:, 0])
    y_pred = clf.predict(data.test_x)[:, None]
    return dde.metrics.get("MAPE")(data.test_y, y_pred)


def mfgp(data):
    
    from mfgp import LinearMFGP

    model = LinearMFGP(noise=0, n_optimization_restarts=5)
    model.train(data.X_lo_train, data.y_lo_train, data.X_hi_train, data.y_hi_train)
    _, _, y_pred, _ = model.predict(data.X_hi_test)
    return dde.metrics.get("MAPE")(data.y_hi_test, y_pred)


def nn(data):
    
    layer_size = [data.train_x.shape[1]] + [32] * 2 + [1]
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]

    loss = "MAPE"
    optimizer = "adam"
    if data.train_x.shape[1] == 3:
        lr = 0.0001
    else:
        lr = 0.001
    epochs = 30000
    
    net = dde.maps.FNN(
        layer_size, activation, initializer, regularization=regularization
    )
    
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss=loss, metrics=["MAPE"])
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return train_state.best_metrics[0]

'''
def validation_model(yname, train_size):
    
    datafem = FEMData(yname, [70])

    mape = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))

        datamodel = ModelData(yname, train_size, "forward")
        X_train, X_test = datamodel.X, datafem.X
        y_train, y_test = datamodel.y, datafem.y

        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        # mape.append(svm(data))
        mape.append(nn(data))

    with open('Output.txt', 'a') as f:
        f.write("model " + yname + ' ' + str(train_size) + str(np.mean(mape, axis=0)) + str(np.std(mape, axis=0)) + '\n')
    print(yname, train_size)
    print(np.mean(mape), np.std(mape))


def validation_FEM(yname, angles, train_size):
    
    datafem = FEMData(yname, angles)
    # datafem = BerkovichData(yname)

    if train_size == 80:
        kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)
    elif train_size == 90:
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
    else:
        kf = ShuffleSplit(
            n_splits=10, test_size=len(datafem.X) - train_size, random_state=0
        )

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datafem.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter))

        X_train, X_test = datafem.X[train_index], datafem.X[test_index]
        y_train, y_test = datafem.y[train_index], datafem.y[test_index]

        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        mape.append(dde.utils.apply(nn, (data,)))

    print(mape)
    print(yname, "validation_FEM ", train_size, ' ', np.mean(mape), ' ', np.std(mape))
    with open('Output.txt', 'a') as f:
        f.write("validation_FEM " + yname + ' ' + str(train_size) + ' ' + str(np.mean(mape, axis=0)) + ' ' + str(np.std(mape, axis=0)) + '\n')

'''

def mfnn(data):
    
    x_dim, y_dim = 3, 1
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]
    net = dde.maps.MfNN(
        [x_dim] + [128] * 2 + [y_dim],
        [8] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
        residue=True,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, loss="MAPE", metrics=["MAPE", "APE SD"])
    losshistory, train_state = model.train(epochs=30000)
    # checker = dde.callbacks.ModelCheckpoint(
    #     "model/model.ckpt", verbose=1, save_better_only=True, period=1000
    # )
    # losshistory, train_state = model.train(epochs=30000, callbacks=[checker])
    # losshistory, train_state = model.train(epochs=5000, model_restore_path="model/model.ckpt-28000")

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return (
        train_state.best_metrics[1],
        train_state.best_metrics[3],
        train_state.best_y[1],
    )

'''
def validation_mf(yname, train_size):
    
    datalow = FEMData(yname, [70])
    # datalow = ModelData(yname, 10000, "forward_n")
    datahigh = BerkovichData(yname)
    # datahigh = FEMData(yname, [70])

    kf = ShuffleSplit(
        n_splits=10, test_size=len(datahigh.X) - train_size, random_state=0
    )
    # kf = LeaveOneOut()

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datahigh.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter), flush=True)

        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=datahigh.X[train_index],
            y_lo_train=datalow.y,
            y_hi_train=datahigh.y[train_index],
            X_hi_test=datahigh.X[test_index],
            y_hi_test=datahigh.y[test_index],
            standardize=True
        )
        mape.append(dde.utils.apply(mfnn, (data,))[0])
        # mape.append(dde.utils.apply(mfgp, (data,)))

    with open('Output.txt', 'a') as f:
        f.write("mf " + yname + ' ' + str(train_size) + ' ' + str(np.mean(mape, axis=0)) + ' ' + str(np.std(mape, axis=0)) + '\n')
    print(mape)
    print(yname, "validation_mf ", train_size, np.mean(mape), np.std(mape))


def validation_scaling(yname):
    
    datafem = FEMData(yname, [70])
    # dataexp = ExpData(yname)
    dataexp = BerkovichData(yname, scale_c=True)

    mape = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))
        data = dde.data.DataSet(
            X_train=datafem.X, y_train=datafem.y, X_test=dataexp.X, y_test=dataexp.y
        )
        mape.append(nn(data))

    print(yname, "validation_scaling", np.mean(mape), np.std(mape))
    with open('Output.txt', 'a') as f:
        f.write("scaling " + yname + ' ' + str(np.mean(mape, axis=0)) + ' ' + str(np.std(mape, axis=0)) + '\n')


def validation_exp(yname, exp, fac=1, typ='err'):
    
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/" + exp + ".csv", yname)

    if fac != 1:
        dataexp.y *= fac

    ape = []
    y = []
    for iter in range(10):
        print("\nIteration: {}".format(iter))
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=dataBerkovich.X,
            y_lo_train=datalow.y,
            y_hi_train=dataBerkovich.y,
            X_hi_test=dataexp.X,
            y_hi_test=dataexp.y,
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname, "validation_exp", np.mean(ape, axis=0), np.std(ape, axis=0))
    if typ == 'n':
        with open('Output.txt', 'a') as f:
            f.write("exp raw " + exp + ' ' + str(fac) + ' ' + yname + ' [' + str(np.mean(y)) + ' ' + str(np.std(y)) + ']\n')
    else:
        with open('Output.txt', 'a') as f:
            f.write("exp " + exp + ' ' + str(fac) + ' ' + yname + ' ' + str(np.mean(ape, axis=0)) + ' ' + str(np.std(ape, axis=0)) + '\n')
    print("Saved to ", yname, ".dat.")
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross(yname, tip, train_size):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/B3067.csv", yname)
    train_size = 10

    ape = []
    y = []

    # cases = range(6)
    # for train_index in itertools.combinations(cases, 3):
    #     train_index = list(train_index)
    #     test_index = list(set(cases) - set(train_index))

    kf = ShuffleSplit(
        n_splits=10, test_size=len(dataexp.X) - train_size, random_state=0
    )
    for train_index, test_index in kf.split(dataexp.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index, "==>", test_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp.X[train_index])),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp.y[train_index])),
            X_hi_test=dataexp.X[test_index],
            y_hi_test=dataexp.y[test_index],
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    with open('Output.txt', 'a') as f:
        f.write("cross " + yname + ' ' + str(np.mean(ape, axis=0)) + str(np.std(ape, axis=0)) + '\n')
    print(yname, "validation_exp_cross", train_size, np.mean(ape, axis=0), np.std(ape, axis=0))
    print("Saved to ", yname, ".dat.")
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross2(yname, train_size, data1, data2, fac=1, typ='err'):
    
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp1 = ExpData("../data/" + data1 + ".csv", yname)
    dataexp2 = ExpData("../data/" + data2 + ".csv", yname)

    ape = []
    y = []

    if fac != 1:
        dataexp1.y *= fac
        dataexp2.y *= fac

    ''''''
    Shufflesplit trains the neural network. train_size is the proportion of the \
        data (0-1) used to train the neural netweork. n_splits is the number of \
        iterations of the training.
    ''''''
    kf = ShuffleSplit(n_splits=10, train_size=train_size, random_state=0)
    ''''''
    This function cycles through the training data output from ShuffleSplit. It \
        displays the training index and records the y values in a .dat file. The \
        mean and standard deviation  
    ''''''
    for train_index, _ in kf.split(dataexp1.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp1.X[train_index])),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp1.y[train_index])),
            X_hi_test=dataexp2.X,
            y_hi_test=dataexp2.y,
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname, "validation_exp_cross2", train_size, np.mean(ape, axis=0), np.std(ape, axis=0))
    if typ == 'n':
        with open('Output.txt', 'a') as f:
            f.write("cross2 raw " + data1 + ' ' + data2 + yname + ' ' + str(fac) + ' ' + str(train_size) + ' [' + str(np.mean(y)) + ' ' + str(np.std(y)) + ']\n')
    else:
        with open('Output.txt', 'a') as f:
            f.write("cross2 " + data1 + ' ' + data2 + yname + ' ' + str(fac) + ' ' + str(train_size) + str(np.mean(ape, axis=0)) + str(np.std(ape, axis=0)) + '\n')
    print("Saved to ", yname, ".dat.")
    np.savetxt(yname + ".dat", np.hstack(y).T)


def validation_exp_cross3(yname, numitrs, expdata):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    if (expdata == "Al6061"):
        dataexp1 = ExpData("../data/Al6061.csv", yname)
        dataexp2 = ExpData("../data/Al6061.csv", yname)
    elif (expdata == "Al7075"):
        dataexp1 = ExpData("../data/Al7075.csv", yname)
        dataexp2 = ExpData("../data/Al7075.csv", yname)
    else:
        dataexp1 = ExpData("../data/Al6061.csv", yname)
        dataexp2 = ExpData("../data/Al7075.csv", yname)

    ape = []
    y = []
    for _ in range(numitrs): 
        print("\nIteration: {}".format(len(ape)))
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=np.vstack((dataBerkovich.X, dataexp1.X)),
            y_lo_train=datalow.y,
            y_hi_train=np.vstack((dataBerkovich.y, dataexp1.y)),
            X_hi_test=dataexp2.X,
            y_hi_test=dataexp2.y,
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    print(yname, "validation_exp_cross3", np.mean(ape, axis=0), np.std(ape, axis=0))
    with open('Output.txt', 'a') as f:
        f.write("cross3 " + yname + ' ' + str(np.mean(ape, axis=0)) + str(np.std(ape, axis=0)) + '\n')
    print("Saved to ", yname, ".dat.")
    print("Saved to  y.dat.")
    np.savetxt("y.dat", np.hstack(y))


def validation_exp_cross_transfer(yname, train_size, dataset):
    datalow = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    dataexp = ExpData("../data/" + dataset + ".csv", yname)
    
    #train_size = 5

    data = dde.data.MfDataSet(
        X_lo_train=datalow.X,
        X_hi_train=dataBerkovich.X,
        y_lo_train=datalow.y,
        y_hi_train=dataBerkovich.y,
        X_hi_test=dataexp.X,
        y_hi_test=dataexp.y,
        standardize=True
    )
    res = dde.utils.apply(mfnn, (data,))
    

    ape = []
    y = []

    kf = ShuffleSplit(
        n_splits=10, test_size=len(dataexp.X) - train_size, random_state=0
    )
    for train_index, test_index in kf.split(dataexp.X):
        print("\nIteration: {}".format(len(ape)))
        print(train_index, "==>", test_index)
        data = dde.data.MfDataSet(
            X_lo_train=datalow.X,
            X_hi_train=dataexp.X[train_index],
            y_lo_train=datalow.y,
            y_hi_train=dataexp.y[train_index],
            X_hi_test=dataexp.X[test_index],
            y_hi_test=dataexp.y[test_index],
            standardize=True
        )
        res = dde.utils.apply(mfnn, (data,))
        ape.append(res[:2])
        y.append(res[2])

    with open('Output.txt', 'a') as f:
        f.write("crosstrans " + dataset  + yname + ' ' + str(train_size) + str(np.mean(ape, axis=0)) + str(np.std(ape, axis=0)) + '\n')
    print(yname)
    print(np.mean(ape, axis=0), np.std(ape, axis=0))
    np.savetxt(yname + ".dat", np.hstack(y).T)

''''''
validation_joe(yname, tsize_1, tsize_2, data_train1, data_train2, data_test)
''''''
def validation_joe(yname, tsize_1, tsize_2, train1_name, train2_name, test_name):
    
    dataFEM = FEMData(yname, [70])
    dataBerkovich = BerkovichData(yname)
    data_train1 = ExpData('../data/' + train1_name + '.csv', yname)
    data_train2 = ExpData('../data/' + train2_name + '.csv', yname)
    data_test = ExpData('../data/' + test_name + '.csv', yname)
    
    ape = []
    y = []
    
    kf = ShuffleSplit(n_splits=10, train_size=tsize_1, random_state=0)
    
    if tsize_1 > 0:
        for train_index, _ in kf.split(data_train1.X):
            print("\nIteration: {}".format(len(ape)))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=dataFEM.X,
                X_hi_train=np.vstack((dataBerkovich.X, data_train1.X[train_index])),
                y_lo_train=dataFEM.y,
                y_hi_train=np.vstack((dataBerkovich.y, data_train1.y[train_index])),
                X_hi_test=data_test.X,
                y_hi_test=data_test.y,
                standardize=True
            )
            res = dde.utils.apply(mfnn, (data,))
            ape.append(res[:2])
            y.append(res[2])
    else:
        for train_index in range(10):
            print("\nIteration: {}".format(train_index))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=dataFEM.X,
                X_hi_train=dataBerkovich.X,
                y_lo_train=dataFEM.y,
                y_hi_train=dataBerkovich.y,
                X_hi_test=data_test.X,
                y_hi_test=data_test.y,
                standardize=True
            )
            res = dde.utils.apply(mfnn, (data,))
            ape.append(res[:2])
            y.append(res[2])
 
    num = 5
    y_res = [res[0]]
    y_data = np.vstack((dataBerkovich.y, data_train1.y[train_index]))
    X_data = np.vstack((dataBerkovich.X, data_train1.X[train_index]))
    find_index = np.argmax(y_data > y_res)
    X_res = X_data[find_index]
    y_res = np.repeat([y_res], num, axis = 0)
    X_res = np.repeat([X_res], num, axis = 0)

    kf = ShuffleSplit(n_splits=10, train_size=tsize_2, random_state=0)

    if tsize_2 > 0:
        for train_index, _ in kf.split(data_train2.X):
            print("\nIteration: {}".format(len(ape)))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=dataBerkovich.X,
                X_hi_train=np.vstack((X_res, data_train2.X[train_index])),
                y_lo_train=dataBerkovich.y,
                y_hi_train=np.vstack((y_res, data_train2.y[train_index])),
                X_hi_test=data_test.X,
                y_hi_test=data_test.y,
                standardize=True
            )
            res = dde.utils.apply(mfnn, (data,))
            ape.append(res[:2])
            y.append(res[2])
    else:
        y_res = np.vstack((y_res, y_res))
        X_res = np.vstack((X_res, X_res))
        print('dataBerkovich.X: ', dataBerkovich.X)
        print('X_res: ', X_res)
        print('dataBerkovich.y: ', dataBerkovich.y)
        print('y_res: ', y_res)
        print('data_test.X: ', data_test.X)
        print('data_test.y: ', data_test.y)
        for train_index in range(10):
            print("\nIteration: {}".format(train_index))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=dataBerkovich.X,
                X_hi_train=X_res,
                y_lo_train=dataBerkovich.y,
                y_hi_train=y_res,
                X_hi_test=data_test.X,
                y_hi_test=data_test.y,
                standardize=True
            )
            res = dde.utils.apply(mfnn, (data,))
            ape.append(res[:2])
            y.append(res[2])

    res = dde.utils.apply(mfnn, (data,))
    print(res)
    
    print(yname, "validation_joe", tsize_1, ' ', tsize_2, np.mean(ape, axis=0), np.std(ape, axis=0))
    with open('Output.txt', 'a') as f:
        f.write('joe ' + train1_name + ' ' + train2_name + ' ' + test_name + yname + ' ' + str(tsize_1) + '-' + str(tsize_2) + 
                str(np.mean(ape, axis=0)) + str(np.std(ape, axis=0)) + '\n')
    print("Saved to ", yname, ".dat.")
    np.savetxt(yname + ".dat", np.hstack(y).T)

def joenn(data, T):
    ''''''
    This is similar to the mfnn function, with an added temperature input.
    ''''''
    x_dim, y_dim = 4, 1
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]
    net = dde.maps.MfNN(
        [x_dim] + [128] * 2 + [y_dim],
        [8] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
        residue=True,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, loss="MAPE", metrics=["MAPE", "APE SD"])
    losshistory, train_state = model.train(epochs=30000)
    # checker = dde.callbacks.ModelCheckpoint(
    #     "model/model.ckpt", verbose=1, save_better_only=True, period=1000
    # )
    # losshistory, train_state = model.train(epochs=30000, callbacks=[checker])
    # losshistory, train_state = model.train(epochs=5000, model_restore_path="model/model.ckpt-28000")
    
    T = T.reshape((1, 4))
    estimate = model.predict(T)
    print('Estimate = ', estimate)
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return (
        train_state.best_metrics[1],
        train_state.best_metrics[3],
        train_state.best_y[1],
    )

def validation_temperature(yname, train_size, dataexp, test_names, typ='err', temp=25):
    dataFEM = FEMDataT(yname, [70])
    dataBerkovich = BerkovichDataT(yname)

    dataexp2 = []
    test_names2 = []
    # Stack all experimental data on top of each other
    for file in dataexp:
        dataexp2.append('../data/' + file + '.csv')
    for file in test_names:
        test_names2.append('../data/' + file + '.csv')
    #print(dataexp2[:])
    #print(test_names2[:])
    dataexp2 = ExpDataT(dataexp2, yname)
    datatest2 = ExpDataT(test_names2, yname)

    ape = []
    y = []

    kf = ShuffleSplit(n_splits=10, train_size=train_size, random_state=0)
    
    if train_size > 0:
        for train_index, _ in kf.split(dataexp2.X):
            print("\nIteration: {}".format(len(ape)))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=dataFEM.X,
                X_hi_train=np.vstack((dataBerkovich.X, dataexp2.X[train_index])),
                y_lo_train=dataFEM.y,
                y_hi_train=np.vstack((dataBerkovich.y, dataexp2.y[train_index])),
                X_hi_test=datatest2.X,
                y_hi_test=datatest2.y,
                standardize=True
            )
            vals = np.mean(data.X_hi_test, axis=0)
            vals[3] = temp
            vals = vals.reshape((1, 4))
            res = dde.utils.apply(joenn, (data, vals, ))
            ape.append(res[:2])
            y.append(res[2])
    else:
        for train_index in range(10):
            print("\nIteration: {}".format(train_index))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=dataFEM.X,
                X_hi_train=dataBerkovich.X,
                y_lo_train=dataFEM.y,
                y_hi_train=dataBerkovich.y,
                X_hi_test=datatest2.X,
                y_hi_test=datatest2.y,
                standardize=True
            )
            vals = np.mean(data.X_hi_test, axis=0)
            vals[3] = temp
            vals = vals.reshape((1, 4))
            res = dde.utils.apply(joenn, (data, vals, ))
            ape.append(res[:2])
            y.append(res[2])

    teststring = ' '.join(test_names)
    outstring = ' '.join(dataexp)
    print(yname, "validation_temperature", train_size, np.mean(ape, axis=0), np.std(ape, axis=0))
    if typ == 'n':
        with open('Output.txt', 'a') as f:
            f.write("temperature raw " + str(temp) + ' [' + teststring + '][' + outstring + ']' + str(train_size) + ' [' + str(np.mean(y)) + ' ' + str(np.std(y)) + ']\n')
    else:
        with open('Output.txt', 'a') as f:
            f.write("temperature " + str(temp) + ' [' + teststring + '][' + outstring + ']' + str(train_size) + str(np.mean(ape, axis=0)) + str(np.std(ape, axis=0)) + '\n')
    print("Saved to ", yname, ".dat.")
    np.savetxt(yname + ".dat", np.hstack(y).T)
'''

def nn1(data):
    
    layer_size = [data.train_x.shape[1]] + [32] * 2 + [1]
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]

    loss = "MAPE"
    optimizer = "adam"
    if data.train_x.shape[1] == 3:
        lr = 0.0001
    else:
        lr = 0.001
    epochs = 30000
    
    net = dde.maps.FNN(
        layer_size, activation, initializer, regularization=regularization
    )
    
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss=loss, metrics=["MAPE"])
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return train_state.best_metrics[0]

def validation_mod_exp(yname, train_size, test_name):
    
    test_name1 = '../data/' + test_name + '.csv'
    dataexp = ExpData(test_name1, yname)
    datamodel = ModelData(yname, train_size, 'forward')
    
    ape = []
    y = []

    if train_size > 0:
        kf = ShuffleSplit(n_splits=10, train_size=train_size, random_state=0)
        for train_index, _ in kf.split(dataexp.X):
            print("\nIteration: {}".format(len(ape)))
            print(train_index)
            data = dde.data.MfDataSet(
                X_lo_train=datamodel.X,
                X_hi_train=dataexp.X,
                y_lo_train=datamodel.y,
                y_hi_train=dataexp.y,
                X_hi_test=dataexp.X,
                y_hi_test=dataexp.y,
                standardize=True
            )
            res = dde.utils.apply(mfnn, (data,))
            ape.append(res[:2])
            y.append(res[2])
    else:
        kf = ShuffleSplit(n_splits=10, train_size=10, random_state=0)
        for train_index, _ in kf.split(dataexp.X):
            print("\nIteration: {}".format(len(ape)))
            print(train_index)
            
            data = dde.data.DataSet(
                X_train=datamodel.X,
                y_train=datamodel.y,
                X_test=dataexp.X,
                y_test=dataexp.y
            )
            res = dde.utils.apply(nn, (data,))
            print('res = ', res)
            '''
            data = dde.data.MfDataSet(
                X_lo_train=dataexp.X,
                X_hi_train=datamodel.X,
                y_lo_train=dataexp.y,
                y_hi_train=datamodel.y,
                X_hi_test=dataexp.X,
                y_hi_test=dataexp.y,
                standardize=True
            )
            '''
            #res = dde.utils.apply(mfnn, (data,))
            res = dde.utils.apply(nn, (data,))
            ape.append(res[:2])
            y.append(res[2])
        #res = dde.utils.apply(nn, (data,))
        #print(res)
        y.append(res)
        #ape.append(res[:2])
        #y.append(res[2])



    with open('Output.txt', 'a') as f:
        f.write("mod_exp " + test_name + ' ' + yname + ' ' + str(train_size) + ' ' + str(np.mean(ape, axis=0)) + ' ' + str(np.std(ape, axis=0)) + '\n')
    print(yname, train_size)
    print(np.mean(ape), np.std(ape))


def main(argument=None):

    if argument != None:
        exec(argument)
        
    return


if __name__ == "__main__":
    main()
