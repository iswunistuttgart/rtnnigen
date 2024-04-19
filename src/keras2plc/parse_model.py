import keras
import struct
import numpy as np
import re

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

    def pack_weights_binary(self)-> bytes:
        all_weights = []
        for layer in self.model.layers:
            if "dropout" in layer.name:
                continue
            
            weight_matrix = layer.get_weights()[0].T.flatten().tolist()
            bias = layer.get_weights()[1].T.flatten().tolist()
            all_weights += weight_matrix + bias

        layer_format = 'd'*len(all_weights)
        return struct.pack(layer_format,*all_weights)

    
    def _get_num_layers(self) -> int:
        return np.array([1 for layer in self.model.layers if not "dropout" in layer.name]).sum()
        # counter = 1
        # nnLayers = self.model.layers
        # for layer in nnLayers:
        #     if "dropout" in layer.name:
        #         continue
        #     else:
        #         counter = counter + 1
        # return counter

    def generate_struct_layers(self) -> str:
        """
        generate the text which is used to define the layers in the struct Layers
        """
        context = f"""
                    num_layers : UINT := {self._get_num_layers()};
                    weights : {self.model_name}_LayerWeights;
                    input : Layer := (num_neurons := {self.input_dim});
                   """
        nnLayers = self.model.layers
        max_num_neurons = self.output_dim

        layers_init = []

        for layer_num in range(len(nnLayers)):
            if "dropout" in nnLayers[layer_num].name:
                continue
            if layer_num == len(nnLayers)-1:
                layers_init.append(f"output : Layer := (num_neurons := {self.output_dim}, pointer_weight:= ADR(weights.OutputLayer_weight),pointer_bias:= ADR(weights.OutputLayer_bias) );")
            else:
                layers_init.append(f"""layer_{layer_num+1} : Layer := (num_neurons := {len(nnLayers[layer_num].get_weights()[1])}, activation := act_type.{nnLayers[layer_num].get_config()["activation"]}, pointer_weight:= ADR(weights.HiddenLayers{layer_num+1}_weight),pointer_bias:= ADR(weights.HiddenLayers{layer_num+1}_bias) );\n\n""")
                if len(nnLayers[layer_num].get_weights()[1]) > max_num_neurons:
                    max_num_neurons = len(nnLayers[layer_num].get_weights()[1])

        context += "\n".join(layers_init)
        
        layer_names = ["input"] + [f"layer_{i}" for i in range(self._get_num_layers()-1)] + ["output"]
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

        for layer_num in range(len(nnLayers)):
            if "dropout" in nnLayers[layer_num].name:
                continue
            
            is_input_layer = layer_num == 0
            is_output_layer = layer_num == len(nnLayers)-1

            dim_curr = self.input_dim-1 if is_input_layer  else len(nnLayers[layer_num-1].get_weights()[1])-1
            dim_next = self.output_dim if is_output_layer else len(nnLayers[layer_num].get_weights()[1])-1
            type = self.nn_data_type

            if is_output_layer:
                layer_role = "OutputLayer"
            else:
                layer_role = f"HiddenLayers{layer_num+1}"

            weights_ST_code += f"""{layer_role}_weight : ARRAY[0..{dim_next},0..{dim_curr}] OF {self.nn_data_type};
                                {layer_role}_bias : ARRAY[0..{dim_next}] OF {self.nn_data_type};
                                """

        return clean_indentation(weights_ST_code)
