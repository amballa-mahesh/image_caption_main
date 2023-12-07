from flask import Flask, redirect, url_for,render_template,request
import matplotlib.pyplot as plt
import os
import requests
from src.utils import query_image,API_URL,headers



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
        path = os.path.join('static/images/img_new.jpg')        
        plt.imsave(path,img)        
        output = query_image(path)
        caption = output[0]['generated_text']      
        print(caption)  
        return render_template('index.html',img_path = path,result = caption)
    
    return redirect(back)



if __name__== '__main__':
    app.run(debug=True)
