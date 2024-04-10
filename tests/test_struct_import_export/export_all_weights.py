import struct
import numpy as np
import keras

# load keras model
model = keras.saving.load_model("./tests/test_model.keras")
nnLayers = model.layers

# load weights and bias into a list
datalist = []
for layer in nnLayers:
    while "dropout" in layer.name:
        continue
    weights = layer.get_weights()[0].T.flatten().tolist()
    bias = layer.get_weights()[1].T.flatten().tolist()
    datalist = datalist + weights + bias

# transfer data to binary format
layer_format = 'd'*len(datalist)
weights_binary = struct.pack(layer_format,*datalist)

# write file
f = open("./tests/test_struct_import_export/AllWeights.txt", "wb")
f.write(weights_binary)
f.close()