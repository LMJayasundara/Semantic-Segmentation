#pip install --no-deps tensorflowjs
import tensorflowjs as tfjs

# load json and create model
from keras.models import model_from_json
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model/model.h5")
print("Loaded model from disk")

tfjs.converters.save_keras_model(loaded_model, "convert_model")