import pickle
from pickle import dump, load
from keras.models import  load_model
from src.utils import generate_caption_image
from src.logger import logging


tokenizer_trained = load(open("artifacts/data/tokenizer_vsc.p","rb"))
print('tokenizer loaded...')
logging.info('tokenizer loaded...')


model_trained = load_model('artifacts/models/model_100+3.h5')
print('model trained loaded')
logging.info('model trained loaded')

model_xception = load_model('artifacts/models/model_xception.h5')
print('model xception loaded')
logging.info('model xception loaded')
max_len = 34
vocab_size = 6683

url = 'artifacts/files/Flicker8k_Dataset/96399948_b86c61bfe6.jpg'


generate_caption_image(url,model_xception,model_trained,tokenizer_trained)

  

      