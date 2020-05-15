import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import keras
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from keras.layers import BatchNormalization
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from keras.models import Model
import numpy as np
from keras.models import load_model
import itertools
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Dataset/utkface/csv/Dataset-4C_3S_inv.csv', sep=';')    #carica il dataset csv contenente la ragliatela e altre features

z_data = []
for index, row in dataset.iterrows():   #carica tutti i file immagini del dataset di cui si è riusciti a calcolare la ragatela
    nome_file = str(row[51])
    path_immagine = "Dataset/utkface/image/" + nome_file
    face = cv2.imread(path_immagine)
    face = cv2.resize(face, (32, 32))
    z_data.append(face)
    if (index % 500) == 0:
        print("Ho processato " + str(index) + " elementi")

z = np.squeeze(z_data)
print(z.shape)

z = z.astype('float32')
z /= 255
print(z.shape)

x = dataset.iloc[:, 0:48].values        #assegna ad x la ragnatela che in questa configurazione ha dimensione 64
y = dataset.iloc[:, 49].values          #assegna ad y il target ovvero il sesso

x_train, x_test_temp, y_train, y_test_temp = train_test_split(x, y, test_size=0.2, random_state=0)      #creazione train e test set l'array ragnatela
z_train, z_test_temp, yy_train, yy_test_temp = train_test_split(z, y, test_size=0.2, random_state=0)    #creazione train e test set per le immagini

#model = load_model('image4C_3S_inv.h5') #salva il modello al termine dell'allenamento, in modo da poterlo caricare successivamente senza dover effettuare di nuovo l'allenamento

#il modello si compone di due reti che si concatenano durante la loro esecuzione
inputA = Input(shape=(32, 32, 3))   #rappresenta i dati dell'immagine
inputB = Input(shape=(48,))         #rappresenta la ragnatela

A = Conv2D(64, (3, 3))(inputA)      #il ramo A è identico alla rete neurale convoluzionale sviluppata in precedenza per le sole immagini
A = MaxPooling2D((2, 2))(A)
A = Conv2D(32, (3, 3))(A)
A = MaxPooling2D((2, 2))(A)
A = Flatten()(A)
A = Dense(256, activation="relu")(A)
A = Dense(256, activation="relu")(A)
A = Model(inputs=inputA, outputs=A)
A.summary()

B = Dense(48, activation="relu")(inputB)        #il ramo B prende in input la ragnatela per ogni immagine
B = Model(inputs=inputB, outputs=B)
B.summary()
combined = concatenate([A.output, B.output])    #i due rami si conatenano in un ulteriore rete fully connected

C = Dense(256, activation="relu")(combined)
C = Dense(256, activation="relu")(C)
C = Dense(1, activation="sigmoid")(C)

model = Model(inputs=[A.input, B.input], outputs=C)
model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )

history = model.fit([z_train, x_train], yy_train, epochs=25, batch_size= 4096, validation_split=0.15)
model.save('image4C_3S_inv.h5')                     #salva il modello al termine dell'allenamento, in modo da poterlo caricare successivamente senza dover effettuare di nuovo l'allenamento

preds = model.predict([z_test_temp, x_test_temp])   #esegue il test con i test set creati prima dell'allenamento

preds_bool = (preds > 0.5)
pd.DataFrame(preds_bool)
cm = confusion_matrix(y_test_temp, preds_bool)      #crea la matrice di confuzione in cui verranno riportati i TP, TF, FP, FT
print(cm)
print("Classification report\n%s\n" % (metrics.classification_report(y_test_temp, preds_bool)))

#grafico accuracy
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



