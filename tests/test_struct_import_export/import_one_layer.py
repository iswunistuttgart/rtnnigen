import struct
import numpy as np

with open("./nn.txt", "rb") as f:
    binary_struct_contents = f.read()

print(binary_struct_contents)

expected_dims = [[6,3], [9,6], [1,9], [6,1], [9,1], [1,1]]

expected_dims_products = [d[0]*d[1] for d in expected_dims]
all_dims = np.array(expected_dims_products).sum()
                 

layer_format = 'd'*all_dims
struct_as_layer = list(struct.unpack(layer_format, binary_struct_contents))

# print(struct_as_layer)

weights = []
i = 0
for dims, len in zip(expected_dims, expected_dims_products):
    w_i = np.array(struct_as_layer[i:i+len]).reshape(dims)
    i +=len
    weights.append(w_i)

print(weights)

#weights = np.array(struct_as_layer)
#print(weights)


## to file:


#possibility1 = struct.pack(layer_format, *weights.flatten().tolist())
##possibility2 = weights.flatten().tobytes()

#print(possibility1)
#print(possibility2)