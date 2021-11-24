import random 
import json 
import pickle
import numpy as np 
import nltk 
from nltk.stem import WordNetLemmatizer
import tensorflow
import pyttsx3 
import speech_recognition as us 
import selenium
from selenium import webdriver as wd 
import cv2


from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))

classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model.h5')

roro = pyttsx3.init('sapi5')
voices = roro.getProperty('voices')
roro.setProperty('voice',voices[1].id)

def speak(audio):
    roro.say(audio)
    roro.runAndWait()

def cleanScentence(sentence):
    sentence_words =nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bagwords(sentence):
    sentence_words = cleanScentence(sentence)
    bag = [0]* len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)




def predictclass(sentence):
    bow = bagwords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_TRESHOLD = 0.25
    result = [[i,r]for i, r in enumerate(res) if r > ERROR_TRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in result:
        return_list.append({'intent':classes[r[0]],'probablity':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    listofintents = intents_json['intents']
    for i in listofintents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def google_chrome():
    speak("What do you wanna search for")
    u = us.Recognizer()
    u.energy_treshold = 5000
    with us.Microphone() as source:
            u.parse_treshold =100
            audio = u.listen(source)
            fact =  u.recognize_google(audio,language ='en-us')
            driver = wd.Chrome()
            fact= u.recognize_google(audio,language ='en-us')
            driver.get('https://www.google.com/search?q={}'.format(fact))
    

def musicalhits():
    speak("name your favourite music")
    

def read_news():
    speak("Name a place you're in")
    


speak("Jexi is UP and Running...")



while True:
    loopr = 2 
    for i in range(loopr):
        u = us.Recognizer()
        u.energy_treshold = 5000
        with us.Microphone() as source:
            speak("At your service what you wanna do...")
            u.parse_treshold =100
            audio = u.listen(source)
        try:
            speak("processing...")
            message= u.recognize_google(audio,language ='en-us')
            print(message)
        except Exception as e:
            speak("come again")


        ints= predictclass(message)
        res = get_response(ints, intents)
        speak(res)
        if(message == "turn on the Chrome"):
                google_chrome()
        elif(message == "Play music"):
                musicalhits() 
        elif(message == "read news"):
                read_news()
        elif(message == "quit"):  
            break
        else:
            speak("Awaiting for yoyr response") 
        




