import string
import re
import os
from src.logger import logging
from src.utils import caption_clean,image_names_with_cleaned_captions,model_xception,extract_img_features,tokenized_captions,tokenizer,max_len_captions,get_image_names,create_sequences,data_generator,create_tokenizer
import pickle
from pickle import dump, load
from keras.utils import to_categorical,pad_sequences




file_name = 'artifacts/files/Flickr8k.lemma.token.txt'
file = open(file_name,'r')
text = file.readlines()
file.close()

image_names = []
captions = []
for lines in text:
    image_name,caption = lines.split('\t')
    image_names.append(image_name)
    captions.append(caption.split('.')[0])
    
print(len(captions))
print(len(image_names))
print(image_names[0:1])
cleaned_captions = caption_clean(captions)

image_names_with_cleaned_captions(image_names,cleaned_captions)

image_features = extract_img_features('artifacts/files/Flicker8k_Dataset',model_xception)
dump(image_features, open("artifacts/data/image_features.p","wb"))



all_features = load(open("artifacts/data/image_features.p","rb"))



url = 'artifacts\data\images_names_with_cleaned_caption.txt'

tokenizer = create_tokenizer(url)

dump(tokenizer, open('artifacts/data/tokenizer_vsc.p', 'wb'))

tokenized_caps = tokenized_captions(url)

print(tokenized_caps[0:1])

max_len = max_len_captions(cleaned_captions)
vocab_size = len(tokenizer.word_index) + 1


image_names = get_image_names(url)

features = []
for name in image_names:
  features.append(all_features[name][0])
print(len(features))

[a,b],c = next(data_generator(features,tokenized_caps,max_len,vocab_size))

print(a.shape)
print(b.shape)
print(c.shape)

