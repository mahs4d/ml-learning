import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from raw_nn import NeuralNetwork

data = np.loadtxt('./data/diabetes.csv', delimiter=',')
X_data = data[:, 0:8]
print('X_data:', X_data.shape)
Y_data = data[:, 8]
print('Y_data:', Y_data.shape)

# Check the dimension of the sets
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=0)
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

nn = NeuralNetwork()

nn.set_input_layer_size(sz=8)

nn.add_dense_layer(16)
nn.add_dense_layer(12)
nn.add_dense_layer(8)
nn.add_dense_layer(1)

nn.build()

nn.train(x=X_train.T, y=y_train.reshape((1, y_train.shape[0])), epochs=1000, learning_rate=0.001)

y_pred = nn.predict(x=X_test.T)
y_pred = y_pred.reshape((y_pred.shape[1],))
y_pred = np.where(y_pred >= 0.5, 1, 0)

print(accuracy_score(y_test, y_pred))
