import keras
import struct
import numpy as np
import re
import hashlib
from abc import ABC, abstractmethod


def clean_indentation(s: str, indent_str: str = "    "):
    """ clears all spaces before rach line in `s` and indents each line with `indent_str` afterwards.
    
    returns: equally indented multiline string"""
    return re.sub(r"^\s*", indent_str, s, flags=re.MULTILINE)

def get_bytes_hash(binary_weights: bytes) -> bytes:
    """ returns a sha256 hash for a given sequence of bytes. """
    m = hashlib.sha256()
    m.update(binary_weights)
    hash_sha_1 = m.digest()
    return hash_sha_1


class model_parser(ABC):
    """ base class to parse dense forward model to general API. """
    def __init__(self, unique_model_name: str, input_dim: int, output_dim: int, has_normalization : bool, has_denormalization: bool):
        self.nn_data_type = "LREAL"
        self.model_name = unique_model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_normalization = has_normalization
        self.has_denormalization = has_denormalization
    
    @abstractmethod
    def _get_all_weights_flattened(self):
        """ returns a sequence of all network weights, matrices are flattened. This is used for weights serialization."""
        pass

    @abstractmethod
    def _get_num_layers(self) -> int:
        """ returns the number of layers of the dense network"""
        pass

    def pack_weights_binary(self) -> bytes:
        """ Packs all network weights to a binary file. """
        all_weights = self.get_all_weights_flattened()

        layer_format = "d" * len(all_weights)
        binary_weights = struct.pack(layer_format, *all_weights)
        binary_weights += get_bytes_hash(binary_weights)
        return binary_weights
    
    def generate_struct_layers(self) -> str:
        """
        generate the text which is used to define the layers in the struct Layers
        """

        num_model_layers = self._get_num_layers()

        normalization_str = ", normalization := act_type.normalization" if self.has_normalization else ""
        context = f"""
                    num_layers : UINT := {num_model_layers+1};
                    weights : {self.model_name}_LayerWeights;
                    layers : ARRAY[0..{num_model_layers}] OF Layer :=[
                    (num_neurons := {self.input_dim}{normalization_str}),
                   """
        #nnLayers = self.model.layers
        # TODO: This is wrong?
        max_num_neurons = self.output_dim

        layers_init = []
        layers_counter = 1
        for layer_num in range(int(self.has_normalization), num_model_layers - int(self.has_denormalization)):
            if "dropout" in nnLayers[layer_num].name:
                continue
            if layer_num == num_model_layers - 1 - int(self.has_denormalization):
                denormalization_add = f"activation := act_type.{nnLayers[layer_num].get_config()['activation']}, {'normalization := act_type.denormalization,' if self.has_denormalization else ''}"
                layers_init.append(
                    f"(num_neurons := {self.output_dim},{denormalization_add} pointer_weight:= ADR(weights.OutputLayer_weight),pointer_bias:= ADR(weights.OutputLayer_bias))"
                )
            else:
                layers_init.append(
                    f"""(num_neurons := {len(nnLayers[layer_num].get_weights()[1])}, activation := act_type.{nnLayers[layer_num].get_config()['activation']}, pointer_weight:= ADR(weights.HiddenLayers{layers_counter}_weight),pointer_bias:= ADR(weights.HiddenLayers{layers_counter}_bias)),"""
                )
                if len(nnLayers[layer_num].get_weights()[1]) > max_num_neurons:
                    max_num_neurons = len(nnLayers[layer_num].get_weights()[1])
                layers_counter += 1

        context += "\n".join(layers_init)
        context += "];\n"
        context = (
            context
            + f"layer_output : ARRAY[0..{max_num_neurons-1}] OF LREAL;\nlayer_input : ARRAY[0..{max_num_neurons-1}] OF {self.nn_data_type};\n"
        )
        return clean_indentation(context)
    


class keras_to_st_parser(model_parser):
    def __init__(self, keras_model: keras.Sequential, unique_model_name: str, input_dim: int, output_dim: int):

        self.model = keras_model
        has_normalization = True if "normalization" in self.model.layers[0].name else False
        has_denormalization = True if "normalization" in self.model.layers[-1].name else False

        super(keras_to_st_parser, self).__init__(unique_model_name, input_dim, output_dim, has_normalization, has_denormalization)


    def _get_all_weights_flattened(self):
        """ returns a sequence of all network weights, matrices are flattened. This is used for weights serialization."""
        all_weights = []
        if self.has_denormalization:
            layers = self.model.layers[int(self.has_normalization) : -1]
        else:
            layers = self.model.layers[int(self.has_normalization) :]

        if self.has_normalization:
            mean = self.model.layers[0].get_weights()[0].T.flatten().tolist()
            std = np.sqrt(self.model.layers[0].get_weights()[1].T.flatten()).tolist()
            all_weights += mean + std
        else:
            mean = [0]
            std = [0]
            all_weights += mean + std

        for layer in layers:
            if "dropout" in layer.name:
                continue

            weight_matrix = layer.get_weights()[0].T.flatten().tolist()
            bias = layer.get_weights()[1].T.flatten().tolist()
            all_weights += weight_matrix + bias

        if self.has_denormalization:
            mean = self.model.layers[-1].get_weights()[0].T.flatten().tolist()
            std = np.sqrt(self.model.layers[-1].get_weights()[1].T.flatten()).tolist()
            all_weights += mean + std
        else:
            mean = [0]
            std = [0]
            all_weights += mean + std

        return all_weights

    def _get_num_layers(self) -> int:
        return np.array([1 for layer in self.model.layers if "dense" in layer.name]).sum()

 

    def generate_struct_layer_weights(self) -> str:
        """
        generate the text which is used to define the matrix in the struct LayerWeights
        """

        weights_ST_code = ""
        nnLayers = self.model.layers

        if self.has_normalization:
            weights_ST_code += f"""
                                normalization_mean : ARRAY[0..{self.input_dim-1}] OF {self.nn_data_type};
                                normalization_std : ARRAY[0..{self.input_dim-1}] OF {self.nn_data_type};
                                """
        else:
            weights_ST_code += f"""
                                normalization_mean : ARRAY[0..0] OF {self.nn_data_type};
                                normalization_std : ARRAY[0..0] OF {self.nn_data_type};
                                """
        layers_counter = 1
        for layer_num in range(len(nnLayers) - int(self.has_denormalization)):

            if "dropout" in nnLayers[layer_num].name:
                continue

            is_input_layer = layer_num == 0
            is_output_layer = layer_num == len(nnLayers) - int(self.has_denormalization) - 1

            if is_input_layer and self.has_normalization:
                continue
            try:
                dim_curr = self.input_dim - 1 if is_input_layer else len(nnLayers[layer_num - 1].get_weights()[1]) - 1
            except:
                dim_curr = self.input_dim - 1 if is_input_layer else len(nnLayers[layer_num - 2].get_weights()[1]) - 1
            dim_next = self.output_dim - 1 if is_output_layer else len(nnLayers[layer_num].get_weights()[1]) - 1

            if is_output_layer:
                layer_role = "OutputLayer"
            else:
                layer_role = f"HiddenLayers{layers_counter}"

            weights_ST_code += f"""{layer_role}_weight : ARRAY[0..{dim_next},0..{dim_curr}] OF {self.nn_data_type};
                                {layer_role}_bias : ARRAY[0..{dim_next}] OF {self.nn_data_type};
                                """
            layers_counter += 1

        if self.has_denormalization:
            weights_ST_code += f"""
                                denormalization_mean : ARRAY[0..{self.output_dim-1}] OF {self.nn_data_type};
                                denormalization_std : ARRAY[0..{self.output_dim-1}] OF {self.nn_data_type};
                                """
        else:
            weights_ST_code += f"""
                                denormalization_mean : ARRAY[0..0] OF {self.nn_data_type};
                                denormalization_std : ARRAY[0..0] OF {self.nn_data_type};
                                """

        weights_ST_code += """hash_sha_256 : ARRAY[0..3] OF LREAL;"""
        return clean_indentation(weights_ST_code)
