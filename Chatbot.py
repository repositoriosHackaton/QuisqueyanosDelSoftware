import random
import json
import pickle
import numpy as np
import unicodedata

import nltk
from nltk.stem import WordNetLemmatizer

from keras import models 
lemmatizer = WordNetLemmatizer()

# Cargar los archivos generados anteriormente
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = models.load_model('chatbot_model.h5')

# Normalizar el texto para manejar correctamente los acentos
def normalize_text(text):
    return unicodedata.normalize('NFKD', text)

# Pasar las palabras de la oración a su forma raíz
def clean_up_sentence(sentence):
    sentence = normalize_text(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertir la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecir la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

# Obtener una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result

def respuesta(message):
    ints = predict_class(message)
    print (message)
    res = get_response(ints, intents)
    return res

while True:
    message = input()
    if message != "":
        print(respuesta(message))
