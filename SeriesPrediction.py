###########################################################################################################################
###################################### Programmer: MEng. Morteza Hajitabar Firuzjaei ######################################
###########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataset = read_csv('./timeseries.csv').values.astype('float32')
print(dataset.shape)

def plot_delta(data):
    plt.plot(data)
    plt.ylabel('close')
    plt.show()

plot_delta(dataset)

def get_y_from_generator(gen):
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    print(y.shape)
    return y

# normalize
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 3

train_data_gen = TimeseriesGenerator(train, train, length=look_back, sampling_rate=1,stride=1, batch_size=3)
test_data_gen = TimeseriesGenerator(test, test, length=look_back, sampling_rate=1,stride=1, batch_size=3)

model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png')
history = model.fit_generator(train_data_gen, epochs=100).history

model.evaluate_generator(test_data_gen)

trainPredict = model.predict_generator(train_data_gen)
testPredict = model.predict_generator(test_data_gen)

trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

trainY = get_y_from_generator(train_data_gen)
testY = get_y_from_generator(test_data_gen)

trainY = scaler.inverse_transform(trainY)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

dataset = scaler.inverse_transform(dataset)

# shift train predictions
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
trainPredictPlot = trainPredictPlot + dataset[0:]

# shift test predictions
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

# Delta + previous close
testPredictPlot = testPredictPlot + dataset[0:]

plt.plot(dataset + dataset[0:])
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
