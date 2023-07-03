#!/usr/bin/env python
# coding: utf-8

# ## INSTALLING ALL THE REQUIRED PACKAGES

# In[1]:


get_ipython().system('pip install --upgrade google-cloud-videointelligence')


# In[2]:


get_ipython().system('pip install --upgrade google-cloud-storage')


# In[3]:


from google.cloud import videointelligence


# In[4]:


import wave, math, contextlib
import speech_recognition as sr
from moviepy.editor import AudioFileClip


# In[5]:


import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[9]:


from gensim.parsing.preprocessing import remove_stopwords


# In[10]:


get_ipython().system('pip install yake')


# In[11]:


import yake


# In[12]:


get_ipython().system('pip install summa')


# In[13]:


from summa import keywords


# In[14]:


import os


# In[15]:


def key_path():
    inp=input("Please enter the key file in json format: ") #Take key json file name as an input from the user
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=inp


# In[16]:


key_path()


# In[17]:


from google.cloud import storage


# # Connecting to the GC bucket and accessing the items in the bucket.

# In[18]:


def bucket():
    global storage_client
    storage_client=storage.Client()
    for bucket in storage_client.list_buckets():   #Listing all present buckets on GCP 
        print(bucket)     


# In[19]:


bucket()


# In[20]:


def objects(bucket_name):
    bucket=storage_client.get_bucket(bucket_name)
    for obj in bucket.list_blobs():  #Listing all objects present in the bucket
        print(obj.name)


# In[21]:


import io


# In[22]:



class my_dictionary(dict):     #Initialize dictionary to store Logos
    # __init__ function 
        def __init__(self): 
            self = dict()   
    # Function to add key:value 
        def add(self, key, value): 
            self[key] = value 

    
def detect_logo_gcs(input_uri): #Detecting all the logos present in the video
    global logo_dictionary
    logo_dictionary = my_dictionary() 
    client = videointelligence.VideoIntelligenceServiceClient()

    features = [videointelligence.Feature.LOGO_RECOGNITION]

    operation = client.annotate_video(
        request={"features": features, "input_uri": input_uri}
    )

    print("Waiting for operation to complete...")
    response = operation.result()
    # Get the first response, since we sent only one video.
    annotation_result = response.annotation_results[0]
    
    # Annotations for list of logos detected, tracked and recognized in video.
    for logo_recognition_annotation in annotation_result.logo_recognition_annotations:
        entity = logo_recognition_annotation.entity

        # Opaque entity ID. Some IDs may be available in [Google Knowledge Graph
        # Search API](https://developers.google.com/knowledge-graph/).
        print("Entity Id : {}".format(entity.entity_id))

        print("Description : {}".format(entity.description))

        # All logo tracks where the recognized logo appears. Each track corresponds
        # to one logo instance appearing in consecutive frames.
        #for track in logo_recognition_annotation.tracks:

            # Video segment of a track.
            
           # print("\tConfidence : {}".format(track.confidence))

            
        # All video segments where the recognized logo appears. There might be
        # multiple instances of the same logo class appearing in one VideoSegment.
        count=0
        for segment in logo_recognition_annotation.segments:
            count=count+1
           # print(
            #    "\n\tStart Time Offset : {}.{}".format(
             #       segment.start_time_offset.seconds,
              #      segment.start_time_offset.microseconds * 1000,
               # )
            #)
            #print(
             #   "\tEnd Time Offset : {}.{}".format(
              #      segment.end_time_offset.seconds,
               #     segment.end_time_offset.microseconds * 1000,
                #)
            #)
            
        logo_dictionary.add(count,entity.description)
        print(logo_dictionary)


# In[23]:


def detect_logo(input_videoname):  #Detecting the logo which is occcuring more often in video
    global logo
    detect_logo_gcs(input_videoname)
    myKeys = list(logo_dictionary.keys())
    myKeys.sort(reverse = True)
    logo=logo_dictionary[myKeys[0]]
    print("Logo of video: ", logo)


# In[24]:


def speech_to_text(x): #Converting video speech to text  
    global text
    text=[]
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]

    config = videointelligence.SpeechTranscriptionConfig(
        language_code="en-US", enable_automatic_punctuation=True
    )
    video_context = videointelligence.VideoContext(speech_transcription_config=config)

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": x,
            "video_context": video_context,
        }
    )

#print("\nProcessing video for speech transcription.")

    result = operation.result(timeout=600)

    annotation_results = result.annotation_results[0]
    for speech_transcription in annotation_results.speech_transcriptions:

        for alternative in speech_transcription.alternatives:
            #print("Transcript: {}".format(alternative.transcript))
            #print("Confidence: {}\n".format(alternative.confidence))
            text.append(alternative.transcript)
        


# In[25]:


def listToString(instr):
    emptystr=""

    for i in instr: 

        emptystr += i+' '
    return emptystr


# In[26]:


def speech_to_txt(input_videoname):  #Converting the video text data to string format
    global text_speech
    speech_to_text(input_videoname)
    text_speech=listToString(text)
    print("Transcript of video: ", text_speech)


# In[27]:


def logo_speech(): 
    global input_videoname
    input_videoname=input("please pass the url of the object in the GCP bucket")
    detect_logo(input_videoname)
    speech_to_txt(input_videoname)


# In[28]:


logo_speech()


# ### In case the API is down seems to have a connecting problem we can use python library to extract the text if the API works fine we can skip this

# In[ ]:


#transcribed_audio_file_name = "transcribed_speech.wav"
#zoom_video_file_name = "2-BHP-Medicine-MoodMatic.mp4"


# In[ ]:


#audioclip = AudioFileClip(zoom_video_file_name)
#audioclip.write_audiofile(transcribed_audio_file_name)


# In[ ]:


#with contextlib.closing(wave.open(transcribed_audio_file_name,'r')) as f:
    #frames = f.getnframes()
    #rate = f.getframerate()
    #duration = frames / float(rate)


# In[ ]:


#total_duration = math.ceil(duration / 150)
#r = sr.Recognizer()


# In[ ]:


# for i in range(0, total_duration):
    #with sr.AudioFile(transcribed_audio_file_name) as source:
     #   audio = r.record(source, offset=i*60, duration=60)
    #f = open("transcription.txt", "a")
    #f.write(r.recognize_google(audio))
    #f.write(" ")
#f.close()


# ### Here we  created a function to remove stopwords and unrequired symbols used genism library which has remove_stopwords method to directly remove the unwanted words.

# In[29]:


def cleaning_text(txt):
    new_data=txt
    temp=[]
    new_data = re.sub(r"http\S+", "", new_data)
    new_data = re.sub("[^a-zA-Z]"," ",new_data) 
    new_data=new_data.lower()
    temp.append(new_data)
    clean_text=temp
    filtered_text = remove_stopwords(clean_text[0])
    return filtered_text


# ### We have the clean_text i.e processed from the ouput received from the speech-text transcription module
# <i> we are going to extract the dominant words from the text the packages used from NLP processing are:</i>
#       
#    1. Yake
#    2. Summa

# In[30]:


def yake_extract(t1):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.3
    numOfKeywords = 10
    custom_kw_extractor = yake.KeywordExtractor(n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords)
    keywords = custom_kw_extractor.extract_keywords(t1)
    for kw in keywords:
        if(kw[1]>=0.10):
            #print(kw[0])
            dominant_words.append(kw[0])


# In[31]:


def summa_extract(t2):
    TR_keywords = keywords.keywords(t2)
    return TR_keywords
    


# <b> We send the text received from the speech to clean the text and then we extract the dominant using <u><i>yake</i></u> and <u><i>summa</i></u> libraries </b>

# In[32]:


def clean_to_dom_words(text_speech):
    global dominant_words
    global dw
    t=cleaning_text(text_speech)
    dominant_words=[]
    yake_extract(t)
    summa_text= summa_extract(t)
    summa_text=summa_text.replace('\n', " ") 
    dominant_words.append(summa_text.rstrip())
    dw=listToString(dominant_words)
    dw=dw.replace(" ",",")
    print("Dominant words of video: ", dw)


# In[33]:


clean_to_dom_words(text_speech)


# ### Model Building
# ###  The models used for training are:
#     1.Naive Bayes
#     2.Decision Tree
#     3.Random Forest 

# In[34]:


def model_building():
    global nb,dt,rf,tfidf
    data=pd.read_csv('company_df.csv')
    X = data[['Entity', 'Description']]
    y = data['Industry_Classification']
    tfidf = TfidfVectorizer()
    tfidf.fit(X['Entity'] + ' ' + X['Description'])
    X_train_transformed_train = tfidf.transform(X['Entity'] + ' ' + X['Description'])
    nb = MultinomialNB()
    nb.fit(X_train_transformed_train, y)
    dt = DecisionTreeClassifier()
    dt.fit(X_train_transformed_train, y)
    rf=RandomForestClassifier(random_state=0)
    rf.fit(X_train_transformed_train, y)


# In[35]:


model_building()


# # Testing the model

# In[36]:


def test_model():
    global y_pred
    data_test=pd.DataFrame({'Logo':[logo],'DominantWords':[dw]})
    X_2 = data_test[['Logo', 'DominantWords']]
    X_test_transformed_1 = tfidf.transform(X_2['Logo'] + ' ' + X_2['DominantWords'])
    y_pred = rf.predict(X_test_transformed_1)
    cn=["Apparel, Footwear and Accessories","Business and Legal","Education","Electronics and Communication","Food and Beverage"
    ,"Health and Beauty","Home and Real Estate","Insurance","Life and Entertainment","Pharmaceutical and Medical","Politics, Government and Organizations",
    "Restaurants","Retail Stores","Travel","Vehicles"]
    class_name = []
    for i in y_pred:
        class_name.append(cn[i])
    for i in class_name:
        return i


# In[38]:


market=test_model()
print("The Industry classification of the video:", market)

