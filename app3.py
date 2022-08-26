#from app2 import load_image
import streamlit as st
import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image
from PIL import Image

classes={
    0: 'Angry',
    1: 'Contempt',
    2: 'Disgust',
    3: 'Fear',
    4: 'Happy',
    5: 'Sadness',
    6: 'Surprise'
}
model = tf.keras.models.load_model('my_model_gray2.h5')
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# img = tf.io.read_file(file)
# img = tf.image.decode_image(img)

if file is not None:

    st.image(file, use_column_width=True)    

    img = tf.keras.preprocessing.image.load_img(file, target_size=(48,48))
    
    x = image.img_to_array(img)
    x = tf.image.resize(x, [48,48])
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images)

    st.write(pred*100)

    class_names = classes.values()
    class_name = list(class_names)
    pred_cls = class_name[np.argmax(pred)]
    st.write(pred_cls, 100 * np.max(pred),'')