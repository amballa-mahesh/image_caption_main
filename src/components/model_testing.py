import pickle
from pickle import dump, load
from keras.models import  load_model
from src.utils import generate_caption_image


tokenizer_trained = load(open("artifacts/data/tokenizer_vsc.p","rb"))
print('tokenizer loaded...')


model_trained = load_model('artifacts/models/model_colab.h5')
print('model trained loaded')

model_xception = load_model('artifacts/models/model_xception.h5')
print('model xception loaded')
max_len = 34
vocab_size = 6683

url = 'artifacts/files/Flicker8k_Dataset/3725814794_30db172f67.jpg'


generate_caption_image(url,model_xception,model_trained,tokenizer_trained)

  

      