########### IMPORTING MODULES ###########
import speech_recognition as sr
# import wikipedia
# import bs4
# from bs4 import BeautifulSoup
# import requests
# import datetime
import random
import time
from pickle import load
from tensorflow.python.keras.models import load_model
import json
from Fucntions.prediction_func import predict_intent_with_nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

########### load model , tokenizer , label encoder , intents ###########
try:
    with open(r'Data\intents.json') as file:
        intents= json.load(file)
except FileNotFoundError as e:
    print(e)
#loading the tokenizer
try:
    with open(r'Data\tokenizer.pickle',"rb") as file_tockenizer:
        tokenizer=load(file_tockenizer)
except FileNotFoundError as e:
    print(f"the error message is : {e}")
#loading label encoder
try:
    with open(r'Data\label_encoder.pickle',"rb") as enc:
        lbl_encoder=load(enc)
except FileNotFoundError as e:
    print(f"the error message is : {e}")
try:
    # load trained model
    load_model=tf.keras.models.load_model('Data\model_trained.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")


########### Initializing pyttsx3 ###########
import pyttsx3
listening = True
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

#text to speech
def say(audio):
    engine.say(audio)
    engine.runAndWait()

########### speech to text ###########
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.energy_threshold=500
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout=3)
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
            return query.lower()
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return "none" # End of takeCommand() function


########### intent and corresponding response prediction ###########

def wishMe(querry):
    from Fucntions.prediction_func import max_len
    text_len=min(max_len,len(querry))
    predicted_intent=predict_intent_with_nltk(load_model,tokenizer,lbl_encoder,querry,text_len,ps, stop_words)
    if predicted_intent in intents:
        responses = intents[predicted_intent]['responses']
        response = random.choice(responses)
        # print("intent : ",predicted_intent)
        # print("Response : ",response)
        return [predicted_intent,response]


def start():
    try:
        while True:
            query=takeCommand()
            if "candy" in query.lower() :
                list_intent_response = wishMe(query)
                say(list_intent_response[1])
                while True:
                    query=takeCommand()
                    if query is None:
                        #counter
                        continue
                    else:
                        execution(query)   
            if query is None:
                continue
    except  Exception as e:
        print(e)
        
def execution(query):
    predict_intent=wishMe(query)
    done=False
    if "time" in query or "date" in query  or "weather" in query or "temperature" in query: # for time date and weather related info
        x = datetime.datetime.now()
        if "date" in query: #Only local Date
            say("Date is "+x.strftime("%A ")+x.strftime("%d ")+x.strftime("%B ")+x.strftime("%Y ")) # Date
        if "time" in query: #time can be searched for local and international zones   
            url = "https://google.com/search?q=" + query
            request_result = requests.get( url )
            soup = bs4.BeautifulSoup( request_result.text  , "html.parser" ) 
            time = soup.find( "div" , class_='BNeawe' ).text  
            say("time is "+time ) 

        if "weather" in query or "temperature" in query:
            url = "https://www.google.com/search?q="+query
            html = requests.get(url).content
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
            temp_element = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'})
            str1 = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text
            data = str1.split('\n')
            sky = data[1]
            say("Temperature is "+ temp)
            say("Sky Description:  "+ sky)
        return
    elif 'wikipedia' in query:     # Wikipedia search
        say('Searching Wikipedia...')
        query = query.replace("wikipedia", "")
        results = wikipedia.summary(query, sentences=4)
        say("According to Wikipedia")
        say(results)
        return
    elif ("google" in query and "search" in query) or ("google" in query and "how to" in query) or "google" in query:
        google_search(query)
        return
    if predict_intent[0]=="joke" and "joke " in query:
        joke=get_joke()
        if joke:
            say(joke)
            done =True
    if predict_intent[0]=="music"
    
        
if __name__ == "__main__":

    intent_exceptions=["goodbye","gratitude","nothing"]
    while True:
        query = takeCommand()
        if "candy" in query:  #candy -> WakeWord
          list_intent_response = wishMe(query)
          say(list_intent_response[1])
          query=takeCommand()
          get_result(query)
          time.sleep(3)

          # if user wants to say anything else 
          say("Anything else that I can help ?") #Anything else user wants to ask 
          query=takeCommand()
          list_intent_response=wishMe(query)
          if list_intent_response[0] in intent_exceptions: #or execute intent_exceptions if nothing else to ask
              say(list_intent_response[1])
              continue
          get_result(query)