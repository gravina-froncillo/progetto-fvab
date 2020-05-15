import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import optimizers
from sklearn.utils import shuffle
from keras.models import load_model

dataset = pd.read_csv('Dataset/utkface/csv/Dataset-4C_4S_var4.csv', sep=';')
print(dataset)
dataset = shuffle(dataset)
print(dataset)

x = dataset.iloc[:, 0:64].values    #carica in x la lista di array che rappresentano le ragnatele per ogni volto, in questa configurazione ha dimensione 80
y = dataset.iloc[:, 66].values      #carica in y l'etichetta riguardante il sesso
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)    #esegue lo split del dataset in train set e test set

#classifier = load_model('spider4C_4S_var4.h5')                                                  #carica il classificatore già allenato per test successivi all'allenamento

classifier = Sequential()
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu', input_dim = 64 ))
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )

history = classifier.fit(x_train, y_train, batch_size = 4096, nb_epoch = 25, validation_split=0.15)

classifier.save('spider4C_4S_var4.h5')       #salva il modello al termine dell'allenamento, in modo da poterlo caricare successivamente senza dover effettuare di nuovo l'allenamento

y_pred = classifier.predict(x_test)         #esegue il test con il test set creato prima dell'allenamento
y_pred_bool = (y_pred > 0.5)
pd.DataFrame(y_pred_bool)

cm = confusion_matrix(y_test, y_pred_bool)  #crea la matrice di confuzione in cui verranno riportati i TP, TF, FP, FT
print(cm)

print("Classification report\n%s\n" % (metrics.classification_report(y_test, y_pred_bool)))     #stampa i risultati ottenuti

#grafico accuratezza
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#grafico loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()