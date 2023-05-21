from math import *
from pandas import DataFrame
from matplotlib import pyplot as plt
import os
import random
from performance import *
from lstm import *


from data_process import *


def getDifferenceStats(seriesData):
    obs = len(seriesData)
    diff = []
    seriesData = [float(i) for i in seriesData]
    for i in range(1, obs):
        diff.append(seriesData[i] - seriesData[i - 1])
    return min(diff), max(diff), sum(diff) / len(diff)


if __name__ == "__main__":
    train_size = 1
    test_size = 400
    print("Starting Process")
    # dh = DataHandler("../dailydata/forex/EURUSD.csv")
    dh = DataHandler("MICROSOFTv2.csv")
    
    # Creates 2 new columns that are lagged by 1. These columns are
    # the 'features'.
    dh.timeSeriesToSupervised()
    print("ended Process")
    tsvalues = dh.tsdata.values
    mi, ma, avg = getDifferenceStats(tsvalues[:, 3])

    train = tsvalues[:train_size, :]  # used to train lstm
    test = tsvalues[train_size:(train_size + test_size), :]  # get errors for error lstm
    out = tsvalues[(train_size + test_size):, :]  # predict these values with both lstm's
    # compare lstm_v rmse with lstm_v + lstm_e prediction rsme

    outx, outy = out[:, 1], out[:, 3]
    trainx, trainy = train[:, 1], train[:, 3]
    testx, testy = test[:, 1], test[:, 3]
    testy_dates = test[:, 2]

    # vec->2d
    trainy = trainy.reshape(trainy.shape[0], 1)
    trainx = trainx.reshape(trainx.shape[0], 1)
    testx = testx.reshape(testx.shape[0], 1)
    outx = outx.reshape(outx.shape[0], 1)
    trainx = trainx.reshape((trainx.shape[0], 1, trainx.shape[1]))
    testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))
    outx = outx.reshape((outx.shape[0], 1, outx.shape[1]))


    base = MyLSTM(trainx.shape[1], 4,
                  [27, 25, 3, 45],
                  trainy.shape[1], epochs=100, batch_size=7)

    error_model = MyLSTM(testx.shape[1], 1, [43], testx.shape[1], epochs=100, batch_size=7)

    if os.path.isfile('base_weights.h6') and os.path.isfile('error_weights.h6'):
        base.load_model_weights('base_weights.h6')
        error_model.load_model_weights('error_weights.h6')
    else:
        print("\n\nTraining Base Model...")
        base.train(trainx, trainy)
        base.save_model_weights('base_weights.h6')

        # get error data and train error lstm
        yhat = base.predict(testx)
        error = testy - yhat[:, 0]
        mse_b = mse(testy, yhat)

        plt.title("Single LSTM Residuals")
        # plt.plot(yhat, error, 'bs', label="residuals") # no clue which we want
        plt.plot(error, 'bs', label="residuals")
        plt.legend()
        # plt.show()

        # error = output (y) for each input (series value)
        e_trainx = testx
        e_trainy = error.reshape(error.shape[0], 1)

        print("\n\nTraining Error Model...")
        error_model.train(e_trainx, e_trainy)
        error_model.save_model_weights('error_weights.h6')
        
    print()
    print()
    print()
    print(trainx)
    print("Yellow")
    print(testx)
    print(outx)
    print("Yellow2")
    print(outx)
    print("HEHE3")

    # with both models trained, pass in out_x to each prediction
    yhat_v = base.predict(outx[:505, :, :])
    print(len(yhat_v))
    yhat_e = error_model.predict(outx[:505, :, :])
    mape_vector = mape(outy[:505], yhat_v)


    yhat_ve = yhat_v + yhat_e
    rootmape_vectorector = RootMse(outy[:505], yhat_ve)
    print(len(yhat_ve))

    yhat_vr = yhat_v.copy()
    for i in range(len(yhat_vr)):
        yhat_vr[i] += random.uniform(-.2, .2)
    rootMse = RootMse(outy[:505], yhat_vr)
    
    print("HEHE")
    print(yhat_v)
    print("HEHE2")
    
    yhat_range = yhat_v.copy()
    for i in range(len(yhat_range)):
        yhat_range[i] += random.uniform(-ma, ma)
    mse_range = RootMse(outy[:505], yhat_range)

    for i in range(len(outx)):
        
        print("\nHybrid model MAPE:\t\t", mape_vector, "\nHybrid RMSE:\t\t\t", rootmape_vectorector)

    plt.title("LSTM-LSTM vs. Actual")
    plt.plot(outy[:525], linestyle='-', label='actual')
    plt.plot(yhat_ve, linestyle='--', label='hybrid')
    plt.ylabel("Microsoft Stock price")
    plt.xlabel("Date From Jan/2015 -- January/2016")
    plt.legend()
    # plt.show()
