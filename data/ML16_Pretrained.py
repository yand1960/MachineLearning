from keras.applications.vgg16 import VGG16
import keras.models as models
import keras.layers as layers

# используем готовую сетку

model = models.Sequential()
model.add(VGG16(weights=None, input_shape=(12, 10, 3), classes=26))