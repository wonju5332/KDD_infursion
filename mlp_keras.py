from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


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

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

loss_and_metrics = model.evaluate(X_test, y_test)

print('\nloss : {} , test_acc : {}\n'.format(loss_and_metrics[0],loss_and_metrics[1] * 100))

# model.summary()
# model.history()
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(X_test, verbose=2)
Y_pred = np.argmax(Y_pred, axis=1)

for ix in range(5):
    print(ix, confusion_matrix(np.argmax(y_test, axis=1), Y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test, axis=1), Y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd

df_cm = pd.DataFrame(cm, range(5),
                     range(5))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()