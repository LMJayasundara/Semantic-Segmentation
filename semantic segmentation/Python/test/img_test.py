import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array


from keras.models import model_from_json
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights
loaded_model.load_weights("./model/model.h5")
print("Loaded model from disk")

def make_prediction(model,img_path,shape):
    img= img_to_array(load_img(img_path , target_size= shape))/255.
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels

def form_colormap(prediction,mapping):
    h,w = prediction.shape
    color_label = np.zeros((h,w,3),dtype=np.uint8)    
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label

class_map_df = pd.read_csv("class_dict.csv")

class_map = []
for index,item in class_map_df.iterrows():
    class_map.append(np.array([item['r'], item['g'], item['b']]))
    
len(class_map)

image = './test.png'
img_size = 128

pred_label = make_prediction(loaded_model, image, (img_size,img_size,3))
print(pred_label.shape)

pred_colored = form_colormap(pred_label,np.array(class_map))

resize = cv2.resize(pred_colored, (960, 540)) 

cv2.imshow('image window', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()