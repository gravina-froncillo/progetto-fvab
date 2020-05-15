import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler
#import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from keras.layers import BatchNormalization
from joblib import dump, load
from keras import optimizers
import pdb
from sklearn.utils import shuffle
from keras.models import load_model

#carico il dataset
dataset = pd.read_csv('Dataset-5C_4S_inv.csv', sep=';')
print(dataset)
dataset = shuffle(dataset)
print(dataset)
#taglio i 64 settori dalle 3 colonne finali che indicano rispettivamente età, sesso e etnia
X = dataset.iloc[:, 0:80].values
print(X.shape)
#print(np.mean(X))
#X = X - np.mean(X, axis = 0)
#print(np.mean(X))
y = dataset.iloc[:, 81].values      #alla colonna 65 abbiamo il sesso (il nostro obiettivo)
#divido i dati e il risultato(il sesso) per il training e il test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#x_validation, X_test, y_validation, y_test = train_test_split(X_test_temp, y_test_temp, test_size = 0.5, random_state = 0)

#Caricare classificatore
#classifier = load_model('spider5C_4S_inv.h5')
#fully connected ridurre, e aumentare capacità. Togli layer e tutti a 256. Provare batch normalization. Dropout.
classifier = Sequential()
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 80 ))
#classifier.add(BatchNormalization())
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
#classifier.add(BatchNormalization())
#classifier.add(Dropout(0.5))
# adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # sigmoid
# compile the ANN
#sgd = optimizers.SGD(learning_rate=0.0001, nesterov=True)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )
# Fitting the ANN with the training set
history = classifier.fit(X_train, y_train, batch_size = 4096, nb_epoch = 25, validation_split=0.15)
classifier.save('spider5C_4S_inv.h5')
y_pred = classifier.predict(X_test)
#print(pd.DataFrame(y_pred))
#matrix confusion
y_pred_bool = (y_pred > 0.5)
pd.DataFrame(y_pred_bool)
#print(pd.DataFrame(y_test))


cm = confusion_matrix(y_test, y_pred_bool)
print(cm)
#stampa i risultati
print("Classification report\n%s\n" % ( metrics.classification_report(y_test, y_pred_bool)))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()