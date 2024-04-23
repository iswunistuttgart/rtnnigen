import keras
import struct
import numpy as np
import re
import hashlib

def clean_indentation(s : str, indent_str : str = '    '):
    return re.sub(r'^\s*', indent_str, s, flags=re.MULTILINE)

class keras_to_st_parser:
    def __init__(
            self,keras_model : keras.Sequential,
            unique_model_name: str, 
            input_dim : int, 
            output_dim : int
            ):

        self.model = keras_model
        self.nn_data_type = "LREAL"
        self.model_name = unique_model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalization = True if "normalization" in self.model.layers[0].name else False
        self.denormalization = True if "normalization" in self.model.layers[-1].name else False

    def pack_weights_binary(self)-> bytes:
        all_weights = []
        if self.denormalization:
            layers = self.model.layers[int(self.normalization):-1]
        else:
            layers = self.model.layers[int(self.normalization):]

        if self.normalization:
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

        
        if self.denormalization:
            mean = self.model.layers[-1].get_weights()[0].T.flatten().tolist()
            std = np.sqrt(self.model.layers[-1].get_weights()[1].T.flatten()).tolist()
            all_weights += mean + std
        else:
            mean = [0]
            std = [0]
            all_weights += mean + std

        layer_format = 'd'*len(all_weights)
        binary_weights = struct.pack(layer_format,*all_weights)
        binary_weights += self.weights_hash_get(binary_weights)
        return binary_weights

    def weights_hash_get(self, binary_weights : bytes)->bytes:
        m = hashlib.sha256()
        m.update(binary_weights)
        hash_sha_1 = m.digest()
        return hash_sha_1
    def _get_num_layers(self) -> int:
        return np.array([1 for layer in self.model.layers if "dense"in layer.name]).sum() + 1

    def generate_struct_layers(self) -> str:
        """
        generate the text which is used to define the layers in the struct Layers
        """

        normalization_str = ", normalization := act_type.normalization" if self.normalization else ""
        context = f"""
                    num_layers : UINT := {self._get_num_layers()};
                    weights : {self.model_name}_LayerWeights;
                    input : Layer := (num_neurons := {self.input_dim}{normalization_str});
                   """
        nnLayers = self.model.layers
        max_num_neurons = self.output_dim

        layers_init = []
        layers_counter = 1
        for layer_num in range(int(self.normalization),len(nnLayers) - int(self.denormalization)):
            if "dropout" in nnLayers[layer_num].name:
                continue
            if layer_num == len(nnLayers)-1-int(self.denormalization):
                denormalization_add =  f'activation := act_type.{nnLayers[layer_num].get_config()['activation']}, {'normalization := act_type.denormalization,' if self.denormalization else ""}'
                layers_init.append(f"output : Layer := (num_neurons := {self.output_dim},{denormalization_add} pointer_weight:= ADR(weights.OutputLayer_weight),pointer_bias:= ADR(weights.OutputLayer_bias) );")
            else:
                layers_init.append(f"""layer_{layers_counter} : Layer := (num_neurons := {len(nnLayers[layer_num].get_weights()[1])}, activation := act_type.{nnLayers[layer_num].get_config()['activation']}, pointer_weight:= ADR(weights.HiddenLayers{layers_counter}_weight),pointer_bias:= ADR(weights.HiddenLayers{layers_counter}_bias) );""")
                if len(nnLayers[layer_num].get_weights()[1]) > max_num_neurons:
                    max_num_neurons = len(nnLayers[layer_num].get_weights()[1])
                layers_counter += 1

        context += "\n".join(layers_init)
        
        layer_names = ["input"] + [f"layer_{i+1}" for i in range(self._get_num_layers()-2)] + ["output"]
        names_sequence = ", ".join(layer_names)
        context = context + f"\nlayers : ARRAY[0..{self._get_num_layers()-1}] OF Layer :=[{names_sequence}];\n"

        context = context + f"layer_output : ARRAY[0..{max_num_neurons-1}] OF LREAL;\nlayer_input : ARRAY[0..{max_num_neurons-1}] OF {self.nn_data_type};\n"
        return clean_indentation(context)
        

    def generate_struct_layer_weights(self) -> str:
        """
        generate the text which is used to define the matrix in the struct LayerWeights
        """
        
        weights_ST_code = ''
        nnLayers = self.model.layers

        if self.normalization:
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
        for layer_num in range(len(nnLayers)-int(self.denormalization)):

            if "dropout" in nnLayers[layer_num].name:
                continue
            
            is_input_layer = layer_num == 0
            is_output_layer = layer_num == len(nnLayers)-int(self.denormalization)-1

            if is_input_layer and self.normalization:
                continue

            dim_curr = self.input_dim-1 if is_input_layer  else len(nnLayers[layer_num-1].get_weights()[1])-1
            dim_next = self.output_dim-1 if is_output_layer else len(nnLayers[layer_num].get_weights()[1])-1
            type = self.nn_data_type

            if is_output_layer:
                layer_role = "OutputLayer"
            else:
                layer_role = f"HiddenLayers{layers_counter}"

            weights_ST_code += f"""{layer_role}_weight : ARRAY[0..{dim_next},0..{dim_curr}] OF {self.nn_data_type};
                                {layer_role}_bias : ARRAY[0..{dim_next}] OF {self.nn_data_type};
                                """
            layers_counter += 1
            
        if self.denormalization:
            weights_ST_code += f"""
                                denormalization_mean : ARRAY[0..{self.output_dim-1}] OF {self.nn_data_type};
                                denormalization_std : ARRAY[0..{self.output_dim-1}] OF {self.nn_data_type};
                                """
        else:
            weights_ST_code += f"""
                                denormalization_mean : ARRAY[0..0] OF {self.nn_data_type};
                                denormalization_std : ARRAY[0..0] OF {self.nn_data_type};
                                """

        weights_ST_code += """hash_sha_256 : ARRAY[0..7] OF LREAL"""
        return clean_indentation(weights_ST_code)
