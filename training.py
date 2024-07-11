import random
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer #Para pasar las palabras a su forma raíz

#Librerias Para crear la red neuronal:
from keras import models, layers, optimizers

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json', encoding='utf-8').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
# (one hot-encoding):
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)

train_x = []
train_y = []
for i in training:
    train_x.append(i[0])
    train_y.append(i[1])

train_x = np.array(train_x)
train_y = np.array(train_y)

# Dividir los datos en conjuntos de entrenamiento y validación
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Creamos la red neuronal
model = models.Sequential()
model.add(layers.Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
model.add(layers.Dropout(0.5, name="hidden_layer1"))
model.add(layers.Dense(64, name="hidden_layer2", activation='relu'))
model.add(layers.Dropout(0.5, name="hidden_layer3"))
model.add(layers.Dense(len(train_y[0]), name="output_layer", activation='softmax'))

# Creamos el optimizador y lo compilamos
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=110, batch_size=5, validation_data=(val_x, val_y), verbose=1)
models.save_model(model, 'chatbot_model.h5')


# Graficar la precisión de entrenamiento y validación:
plt.figure(figsize=(12, 4))

# Graficar la función de coste:
plt.plot(history.history['loss'])
plt.title('Función de coste durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.savefig('Grafico_coste.png') 
plt.close()  

# Gráfico de precisión de entrenamiento:
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento', color='blue', linestyle='--', marker='o')
plt.title('Precisión de entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.savefig('Grafico_Precision_Entrenamiento.png')
plt.close() 

# Gráfico de precisión de validación:
plt.plot(history.history['val_accuracy'], label='Precisión de validación', color='green', linestyle='-', marker='x')
plt.title('Precisión de validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.savefig('Grafico_precision_Validacion.png')
plt.close()
