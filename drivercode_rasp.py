import subprocess
import speech_recognition as sr
import wikipedia
import bs4
from bs4 import BeautifulSoup
import requests
import datetime
import time
import voice_greetings
#import ubidots_separatecode
################################################
#Initializing pyttsx3
# import pyttsx3
# listening = True
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)

# def say(audio):
#     engine.say(audio)
#     engine.runAndWait()
###########################################
#Initializing  tts
def say(text):
    max_chunk_length = 150  # Set an arbitrary maximum length for each chunk
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    for chunk in chunks:
        command = [
            'mpg123',
            '-q',  # Quiet mode
            f'http://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&q={chunk}&tl=en'
        ]
        subprocess.run(command)
        time.sleep(0.5)  # Add a small delay between chunks to ensure smooth playback

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
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
    
def wishMe(query):
    list_int_resp=voice_greetings.predict_intent(query) 
    return list_int_resp

   
    

def google_search(search_text):
    result = ''
    search_data = search_text
    if "who is" in search_data or "who are" in search_data:
        search_data = search_data.split(" ")[2:]
        search_data = " ".join(search_data)
        try:
            result = wikipedia.summary(search_data, sentences=2)
            return result
        except Exception as e:
            pass
    else:
        url = "https://www.google.co.in/search?q=" + search_data
        try:
            search_result = requests.get(url).text
            soup = BeautifulSoup(search_result, 'html.parser')
            result_div = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')
            result = result_div[0].text
            return result
        
        except Exception as e:
            pass
    return("sorry could not find any information")


def get_result(query):
    if query.lower()=="none":       # query= "tell me about kolkata"
        say("No instructions found")

    elif "time" in query or "date" in query  or "weather" in query or "temperature" in query: # for time date and weather related info
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
        

          
    elif 'wikipedia' in query:     # Wikipedia search
        say('Searching Wikipedia...')
        query = query.replace("wikipedia", "")
        results = wikipedia.summary(query, sentences=4)
        say("According to Wikipedia")
        say(results)

    else:
        results = google_search(query) #google serach
        print(results)
        say(results)
    

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
              
              
            

          
          


