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
from joblib import dump, load
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

#carica il dataset per recuperare il nome delle immagini da processare
#si è scelto di caricare le stesse immagini risultate valide ne pre-processing
dataset = pd.read_csv('Dataset-4C_4S_var4.csv', sep=';')

z_data = []
for index, row in dataset.iterrows():
    nome_file = str(row[67])
    path_immagine = "Dataset/" + nome_file
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


y = dataset.iloc[:, 66].values              #alla 66 colonna è specificato il sesso di ogni volto nelle immagini
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=0)

#model = load_model('image.h5')   #carica il classificatore già allenato per test successivi all'allenamento

inputA = Input(shape=(32, 32, 3))           #rete neurale convoluzionale
A = Conv2D(64, (3, 3))(inputA)
A = MaxPooling2D((2, 2))(A)
A = Conv2D(32, (3, 3))(A)
A = MaxPooling2D((2, 2))(A)
A = Flatten()(A)
A = Dense(256, activation="relu")(A)
A = Dense(256, activation="relu")(A)
A = Dense(1, activation="sigmoid")(A)
model = Model(inputs=inputA, outputs=A)
model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )

history = model.fit(z_train, y_train, epochs=25, batch_size= 4096, validation_split=0.15)
model.save('image.h5')                      #salva il modello al termine dell'allenamento, in modo da poterlo caricare successivamente senza dover effettuare di nuovo l'allenamento

preds = model.predict(z_test)               #esegue il test con il test set creato prima dell'allenamento

preds_bool = (preds > 0.5)
pd.DataFrame(preds_bool)
cm = confusion_matrix(y_test, preds_bool)   #crea la matrice di confuzione in cui verranno riportati i TP, TF, FP, FT
print(cm)
print("Classification report\n%s\n" % (metrics.classification_report(y_test, preds_bool)))      #stampa i risultati ottenuti

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
