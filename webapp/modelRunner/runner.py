#added ModelRunner Auto Encoder Decoder

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join('..', 'img_inpaint')))
# import app

from tensorflow import keras
from .pconv import PConv2D
from tensorflow.keras.models import model_from_json
from PIL import Image


cur_path = os.path.dirname(os.path.abspath(__file__))
print(cur_path)
output_path = os.path.normpath(cur_path+'/../static/uploads/output/')
print(output_path)
input_path = os.path.normpath(cur_path+'/../static/input/')
print(input_path)

json_file = open(cur_path+'/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={'PConv2D':PConv2D})
loaded_model.load_weights(cur_path+"/model.h5")


def clean_input_dir():
    # clean the input sub-dir in static
    filelist = [f for f in os.listdir(input_path)]
    for f in filelist:
        os.remove(os.path.join(input_path, f))
    return

def rgba_to_rgb(image1, image2, color=(255, 255, 255)):
    image1.load()
    image2.load()
    background1, background2 = (Image.new('RGB', image1.size, color), 
                                        Image.new('RGB', image2.size, color))
    background1.paste(image1, mask=image1.split()[3])
    background2.paste(image2, mask=image2.split()[3])
    return (background1.save(os.path.join(input_path,'mi_new.jpg'), 'JPEG', quality=80),
            background2.save(os.path.join(input_path,'m_new.jpg'), 'JPEG', quality=80))

def imginp(masked_image, mask):
    masked_img = Image.open(masked_image)
    mask = Image.open(mask)
    # convert rgba to rgb -- Need to add a check for those images which 
    # either don't have an alpha channel OR have partial data in alpha channel
    rgba_to_rgb(masked_img, mask)

    mi_new = Image.open(os.path.join(input_path,'mi_new.jpg'))
    m_new = Image.open(os.path.join(input_path,'m_new.jpg'))

    mi_new = mi_new.resize((32,32))
    m_new = m_new.resize((32,32))
    mi_r = np.array(mi_new)
    m_r = np.array(m_new)

    inp = [mi_r.reshape((1,) + mi_r.shape), m_r.reshape((1,) + m_r.shape)]
    val = loaded_model.predict(inp)
    pred_img = val.reshape(val.shape[1:])
    fimg = Image.fromarray(pred_img, 'RGB')

    clean_input_dir()
    # return fimg.save(output_path+"image.jpg", 'JPEG', quality=80)
    return fimg.save(os.path.join(output_path,"image.jpg"), 'JPEG', quality=80)



# if __name__ == "__main__":
#     app.make_static_dir()
#     imginp("modelRunner/masked_image.png", "modelRunner/mask.png")
