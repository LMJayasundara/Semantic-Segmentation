import glob
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_model():
    json_file = open('./model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights("./model/model.h5")
    print("Loaded model from disk")
    return loaded_model

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

# def load_class_map():
class_map_df = pd.read_csv("class_dict.csv")
class_map = []
for index,item in class_map_df.iterrows():
    class_map.append(np.array([item['r'], item['g'], item['b']]))

loaded_model = load_model()
# load_class_map()

path = glob.glob("./data/test/*.png")
img_size = 128

# path = ["./data/test/1593516458050407424.png"] 

for img in path:
    pred_label = make_prediction(loaded_model, img, (img_size,img_size,3))

    pred_colored = form_colormap(pred_label,np.array(class_map))
    frame = cv2.resize(pred_colored, (620, 480))

    image = cv2.imread(img)
    framex = cv2.resize(image, (620, 480))

    
    numpy_horizontal = np.hstack((framex, frame))

    cv2.imshow('Frame', numpy_horizontal)
    time.sleep(1)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()