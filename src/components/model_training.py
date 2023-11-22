import string
import re
import os
from src.logger import logging
from src.utils import caption_clean,image_names_with_cleaned_captions,model_xception,extract_img_features,tokenized_captions,tokenizer,max_len_captions,get_image_names,create_sequences,data_generator,define_model
import pickle
from pickle import dump, load
from keras.utils import to_categorical,pad_sequences,plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense,Add,LSTM, Embedding, Dropout,Layer





url = 'artifacts\data\images_names_with_cleaned_caption.txt'

tokenized_caps = tokenized_captions(url)

print(tokenized_caps[0:1])

max_len = 34
vocab_size = 6683


image_names = get_image_names(url)

print(len(tokenized_caps))
print(len(image_names))


all_features = load(open("artifacts/data/image_features.p","rb"))


features = []
for name in image_names:
  features.append(all_features[name][0])
print(len(features))

[a,b],c = next(data_generator(features,tokenized_caps,max_len,vocab_size))

print(a.shape)
print(b.shape)
print(c.shape)

model = define_model(vocab_size,max_len)

epochs = 30
steps = 64

for i in range(epochs):
  generator = data_generator(features, tokenized_caps, max_len, vocab_size)
  model.fit(generator, epochs = 1,verbose = 3, steps_per_epoch=steps)
  print('epochs',i)
  model.save("artifacts/models/model_" +str(i)  + ".h5")

