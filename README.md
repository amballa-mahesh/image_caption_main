<<<<<<< HEAD

Readme file.

This project is Developed on Python_version ==3.8.0


LIVE HOST WEB ADDRESS: https://image-captioning-iwhw.onrender.com

if this host is taking time use the below mentioned site..(Loading speed..)

Alternate Address: https://image-captioning-wine.vercel.app/


This the Image Captioning Model.

Steps involved in creating this model are - 

EDA - 

reading data from the text file using file option (open,read)
data spliting to dependent and independent variables
here the independent are imagenames and captions are to be predicted
captions are cleaned by below transformatios
removing html tags
removing urls
making correction of words
removing punctuations
removing stopwords
making tokenisation keras tokenizer
removing the single and numerical words
form the sequences using pad sequences..
using the keras Xeception model to get the features of the images and stored in the file
Create the inputs and outputs tensors using the image features and sequences(using start and end)


Model Creation-
Using the CNN and RNN model
Keras Xception model is used as the CNN
LSTM layers are used as RNN
this combined model is trained on the inputs and outputs


Prediction:
The Trained CNN + RNN model is used for predictions


Creation of User GUI-

Using the flask library we created the use GUI with HTML and CSS.
Deploy this model in local server.
get the values of the feilds selected by the user by flask
Convert images into features using CNN and process the same to model.
get the predictions(captions) from the model.
return that back to user.

Using the logging

We will write back the logs to the logs.log file

Updating the data to mysql.

from the front end user interface get the values of selected feilds and save them back to local database by python mysql connector, cassandra database.
download the data from the database and share....


=======

