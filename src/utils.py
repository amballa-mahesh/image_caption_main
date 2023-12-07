import string
import re
import os
import tqdm
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical,load_img, img_to_array,pad_sequences
from src.logger import logging
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense,Add
from keras.layers import LSTM, Embedding, Dropout,Layer
import requests

logging.info('libraries loaded')


tokenizer = Tokenizer()

def remove_punctuations(text):    
    return text.translate(str.maketrans('','', string.punctuation))


def word_corrections(text):
    
    text = re.sub(r"didn't", "did not", text)    
    text = re.sub(r"don't", "do not", text)    
    text = re.sub(r"won't", "will not", text)    
    text = re.sub(r"can't", "can not", text)    
    text = re.sub(r"wasn't", "do not", text)    
    text = re.sub(r"should't", "should not", text)    
    text = re.sub(r"could't", "could not", text)    
    text = re.sub(r"\'ve", " have", text)    
    text = re.sub(r"\'m", " am", text)    
    text = re.sub(r"\'ll", " will", text)    
    text = re.sub(r"\'re", " are", text)    
    text = re.sub(r"\'d", " would", text)    
    text = re.sub(r"\'t", " not", text)    
    text = re.sub(r"\'m", " am", text)    
    text = re.sub(r"n\'t", " not", text)    
    return text


def caption_clean(captions):    
    cleaned_captions = []
    for caption in captions:
        caption = caption.lower().strip()        
        caption = word_corrections(caption)        
        caption = remove_punctuations(caption)        
        caption = [word for word in caption.split(' ') if (len(word)>1)]        
        caption = [word for word in caption if (word.isalpha())]        
        caption = ' '.join([word for word in caption])        
        caption = 'start'+' '+caption+' '+'end'        
        cleaned_captions.append(caption) 
    return cleaned_captions



def image_names_with_cleaned_captions(image_names,cleaned_captions):
    str_1 = []
    
    for i,j in zip(image_names,cleaned_captions):
        str_2 = i.split('.')[0]+'.jpg' + '\t' + str(j) + '\n'
        str_1.append(str_2)
    data = ''.join([i for i in str_1])
    file = open('artifacts/data/images_names_with_cleaned_caption.txt','w')
    file.write(data)
    file.close()
    
        
model_xception = Xception(include_top=False, pooling='avg' )
model_xception.save("artifacts/models/model_xception.h5")


def extract_img_features(directory,model):
    features = {}
    images = []
    for pic in tqdm(os.listdir(directory)):        
        file = directory+"/"+pic        
        image = Image.open(file)        
        image = image.resize((299,299))        
        image = np.expand_dims(image,axis =0)        
        image = image/127.5        
        image = image - 1        
        feature = model.predict(image)        
        features[pic] = feature        
    return features


def tokenized_captions(data_url):
  file_name = data_url  
  file = open(file_name,'r')
  text = file.readlines()
  file.close()
  
  captions = []  
  for lines in text:      
    _,caption = lines.split('\t')    
    captions.append(caption[:-1])
  tokenizer.fit_on_texts(captions)  
  tokenized_captions = []  
  for caption in captions:      
    seq = tokenizer.texts_to_sequences([caption])    
    tokenized_captions.append(seq)
  return tokenized_captions



def max_len_captions(captions):
  return max(len(caption.split()) for caption in captions)


def get_image_names(data_url):    
  file_name = data_url  
  file = open(file_name,'r')  
  text = file.readlines()  
  file.close()  
  
  image_names = []  
  for lines in text:      
    image_name,_ = lines.split('\t')    
    image_names.append(image_name)       
  return image_names



def create_sequences(feature,cap,max_len,vocab_size):    
  x_1, x_2, y = list(),list(),list()
  for j in range(1,len(cap[0])):      
    in_seq, out_seq = cap[0][:j],cap[0][j]    
    in_seq  = pad_sequences([in_seq],maxlen = max_len)[0]    
    out_seq = to_categorical([out_seq],num_classes = vocab_size)[0]    
    x_1.append(feature)    
    x_2.append(in_seq)    
    y.append(out_seq)  
  return np.array(x_1),np.array(x_2),np.array(y)


def data_generator(features,caps,max_len,vocab_size):    
    for feature,cap in zip(features,caps):        
      inp_image, inp_seq, op_word = create_sequences(feature,cap,max_len,vocab_size)      
      yield [[inp_image, inp_seq], op_word]
      
      
def get_word(value,tokenizer):    
  for word,index in tokenizer.word_index.items():      
    if index == value:
      return word
  
  
def create_tokenizer(data_url):
  file_name = data_url  
  file = open(file_name,'r')  
  text = file.readlines()  
  file.close()  
  captions = []  
  for lines in text:      
    _,caption = lines.split('\t')    
    captions.append(caption[:-1])    
  tokenizer.fit_on_texts(captions)
  return tokenizer


def generate_desc(model, tokenizer, photo, max_length):    
   caption = 'start'   
   for i in range(max_length):       
       sequence = tokenizer.texts_to_sequences([caption])[0]       
       sequence = pad_sequences([sequence], maxlen = max_length)  
     
       pred = model.predict([photo,sequence], verbose=0)       
       pred = int(np.argmax(pred))       
       word = get_word(pred, tokenizer)              
       if word is None:
        break
       caption += ' ' + str(word)
       if word == 'end':
        break       
       caption = re.sub(r"start", "< ", caption)   
   caption = re.sub(r"end", " >", caption)   
   print(caption)
   return caption


def generate_caption_image(url,model_xception,model_trained,tokenizer):    
    
    image = Image.open(url)    
    image = image.resize((299,299))    
    image = np.expand_dims(image,axis =0)    
    image = image/127.5     
    image = image - 1    
    feature = model_xception.predict(image)[0]    
    feature = np.expand_dims(feature,axis=0)    
    caption = generate_desc(model_trained,tokenizer,feature,max_length=34)       
         
    return caption
  
  
  
def define_model(vocab_size=6683,max_length=34):

  inputs1 = Input(shape = (2048,))
  fe1     = Dropout(0.25)(inputs1)
  fe2     = Dense(256,activation = 'relu')(fe1)
  fe3     = Dense(256,activation = 'relu',)(fe2)


  inputs2 = Input(shape = (max_length,))
  se1     = Embedding(vocab_size, 256 , mask_zero = True )(inputs2)
  se2     = Dropout(0.25)(se1)
  se3     = (LSTM(256))(se2)
  se4     = Dropout(0.25)(se3)


  decoder1 = Add()([fe3,se4])
  decoder2 = Dense(256,activation ='relu')(decoder1)

  outputs = Dense(vocab_size, activation= 'softmax')(decoder2)

  model = Model(inputs = [inputs1,inputs2],outputs= outputs)
  model.compile(loss = 'categorical_crossentropy',optimizer='adam')

  print(model.summary())

  return model

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer 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"}

def query_image(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()
