import keras
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.utils import load_img, img_to_array
from PIL import Image

def image_prediction(img_path):
  # Load vgg16
  vgg16file = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
  vgg16 =  VGG16(include_top=False, input_shape=(224,224,3), weights=vgg16file)
  # Load Keras model
  # model_keras = load_model('CatsDogs_sequentialModel.h5')
  model_keras = load_model("CatsDogs_sequentialModel.h5")

  class_labels = ['Cats', 'Dogs']

  # Preprocess the new image
  new_image = preprocess_image(img_path)

  # Extract features from the new image using VGG16 model
  new_features = vgg16.predict(new_image)

  # Make prediction on the new image
  prediction = model_keras.predict(new_features)
  print(prediction)
  predicted_label = np.argmax(prediction)
  print(predicted_label)
  return class_labels[predicted_label]
  
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# if __name__=='__main__':
#   image_prediction('/Users/jamelbelgacem/Documents/Python/Deep learning/Picture classification/cats and dogs/data/train/dogs/dog_591.jpg')
