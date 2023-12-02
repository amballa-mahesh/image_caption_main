from flask import Flask, redirect, url_for,render_template,request
import logging
from src.logger import logging
from pickle import load
from keras.models import load_model
from src.utils import generate_caption_image,define_model
import matplotlib.pyplot as plt
import os
import PIL
import numpy as np
print('libraries loaded..')
logging.info('libraries loaded..')



tokenizer_trained = load(open("artifacts/data/tokenizer_vsc.p","rb"))
print('tokenizer loaded...')
logging.info('tokenizer loaded...')

model_trained = define_model()
model_trained.load_weights('artifacts/models/model_weights+25.h5')
print('model trained loaded')
logging.info('model trained loaded')

model_xception = load_model('artifacts/models/model_xception.h5')
print('model xception loaded')
max_len = 34
vocab_size = 6683


app = Flask(__name__)

@app.route('/')
def welcome_user():
    return render_template('index.html')


@app.route('/submit', methods = ['POST','GET'])
def submit():
    back = request.referrer
    if request.method == 'POST':
       
        img  = request.files['img'] 
        img = plt.imread(img)
        path = 'static/images/img_new.jpg'
        plt.imsave(path,img) 
            
        caption = generate_caption_image(path,model_xception,model_trained,tokenizer_trained)
        return render_template('index.html',img_path = path,result = caption)
        
        
    return redirect(back)
    
    
    
if __name__== '__main__':
    app.run(debug=True)


