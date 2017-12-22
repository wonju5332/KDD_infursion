from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop


from data_wrangling import *


n_in = len(X_train[0])
n_hidden = 200
n_out = len(y_train[0])

model = Sequential()

model.add(Dense(n_hidden, input_dim=n_in ))   # 122 x 200
model.add(Activation('relu'))

model.add(Dense(400) )   # 200 x 100
model.add(Activation('relu'))

model.add(Dense(n_out))
model.add(Activation('softmax'))


epochs = 10
batch_size = 200

model.compile(loss='binary_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

loss_and_metrics = model.evaluate(X_test, y_test)


y_pred = model.predict_classes(X_test)



#  0s[0.082886821183315335, 0.97239395238655058]  #약 97%의 정확도.
