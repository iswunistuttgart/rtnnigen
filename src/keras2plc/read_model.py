import os
import logging
import keras
import struct
from keras2plc.template_strings import template_Layers_WeightsBias,template_Output_WeightsBias,template_normalization_MeanStd,template_denormalization_MeanStd
import numpy as np

class nn_reader:
    def __init__(
            self,
            model_file_Name : str, 
            unique_model_name: str, 
            ):
        self.path = model_file_Name
        # load keras model
        if not os.path.exists(model_file_Name):
            logging.warning(
                    f"Model '{model_file_Name}' don't exist"
                )
        else:
            self.model = keras.saving.load_model(model_file_Name)
        self.nn_data_type = "LREAL"
        self.model_name = unique_model_name
        self.input_dim = self.model.layers[0].input.shape[1]
        self.output_dim = self.model.layers[-1].output.shape[1]
        self.normalization = True if "normalization" in self.model.layers[0].name else False
        self.denormalization = True if "normalization" in self.model.layers[-1].name else False

    def generate_weights_file(self,foldpath : str, overwrite_if_exists : bool = False):
        """
        Save weights and bias of all layers into a binary files, which can be 
        loaded automatically when neural network is initializing in twincat
        """
        nnLayers = self.model.layers
        if self.normalization:
            nnLayers_dense = nnLayers[1:]
        if self.denormalization:
            nnLayers_dense = nnLayers_dense[:-1]
        # load weights and bias into a list
        datalist = []
        for layer in nnLayers_dense:
            while "dropout" in layer.name:
                continue
            weights = layer.get_weights()[0].T.flatten().tolist()
            bias = layer.get_weights()[1].T.flatten().tolist()
            datalist = datalist + weights + bias
        if self.normalization:
            mean = nnLayers[0].get_weights()[0].T.flatten().tolist()
            std = np.sqrt(nnLayers[0].get_weights()[1].T.flatten()).tolist()
            datalist = datalist + mean + std
        if self.denormalization:
            mean = nnLayers[-1].get_weights()[0].T.flatten().tolist()
            std = np.sqrt(nnLayers[-1].get_weights()[1].T.flatten()).tolist()
            datalist = datalist + mean + std
        # transfer data to binary format
        layer_format = 'd'*len(datalist)
        weights_binary = struct.pack(layer_format,*datalist)
        
        # write file
        file_name = f"AllWeights_{self.model_name}.txt"
        file_path = os.path.join(foldpath, file_name)
        if not overwrite_if_exists and os.path.exists(file_path):
            logging.warning(
                f"File '{file_path}' exists and `generate_weights_file` was not set to overwrite the old contents."
                + "The existing model was not overwritten. Either rename the model of allow overwriting."
            )
        else:
            with open(file_path, "wb") as f:
                f.write(weights_binary)
    def _get_num_layers(self) -> int:
        counter = 1
        nnLayers = self.model.layers
        for layer in nnLayers:
            if "dense" in layer.name:
                counter = counter + 1
            else:
                continue
        return counter

    def generate_struct_layers(self) -> str:
        """
        generate the text which is used to define the layers in the struct Layers
        """
        nnLayers = self.model.layers
        context = f'\nnum_layers : UINT := {self._get_num_layers()};\n'
        context = context + f'weights : {self.model_name}_LayerWeights;\n'

        if self.normalization:
            context = context + f"input : Layer := (num_neurons := {self.input_dim},normalization := act_type.normalization);\n"
            nnLayers = nnLayers[1:]
        else:
            context = context + f"input : Layer := (num_neurons := {self.input_dim});\n"

        if self.denormalization:
            nnLayers = nnLayers[:-1]
        
        pointer = 1
        max_num_neurons = self.output_dim
        for layer_num,layer in enumerate(nnLayers):
            if "dropout" in layer.name:
                continue

            if layer_num == len(nnLayers)-1:
                if self.denormalization:
                    context = context + f"output : Layer := (num_neurons := {self.output_dim}, activation := act_type.{layer.get_config()["activation"]} ,normalization := act_type.denormalization , pointer_weight:= ADR(weights.OutputLayer_weight),pointer_bias:= ADR(weights.OutputLayer_bias) );\n"
                else:
                    context = context + f"output : Layer := (num_neurons := {self.output_dim}, pointer_weight:= ADR(weights.OutputLayer_weight),pointer_bias:= ADR(weights.OutputLayer_bias) );\n"  
            else:
                context = context + f"layer_{pointer} : Layer := (num_neurons := {len(layer.get_weights()[1])}, activation := act_type.{layer.get_config()["activation"]}, pointer_weight:= ADR(weights.HiddenLayers{pointer}_weight),pointer_bias:= ADR(weights.HiddenLayers{pointer}_bias) );\n"
                pointer = pointer + 1
                if len(layer.get_weights()[1]) > max_num_neurons:
                    max_num_neurons = len(layer.get_weights()[1])


        context = context + f"layers : ARRAY[0..{self._get_num_layers()-1}] OF Layer :=["
        for i in range(self._get_num_layers()):
            if i == 0: 
                context = context + "input, "
            elif i == self._get_num_layers() -1:
                context = context + "output];\n"
            else:
                context = context + f"layer_{i}, "

        context = context + f"layer_output : ARRAY[0..{max_num_neurons-1}] OF LREAL;\nlayer_input : ARRAY[0..{max_num_neurons-1}] OF {self.nn_data_type};"
        return context
        

    def generate_struct_layersweights(self) -> str:
        """
        generate the text which is used to define the matrix in the struct LayerWeights
        """
        
        context = ''
        nnLayers = self.model.layers
        if self.normalization:
            nnLayers = nnLayers[1:]
        if self.denormalization:
            nnLayers = nnLayers[:-1]
        pointer = 1
        for layer_num in range(len(nnLayers)):
            if "dropout" in nnLayers[layer_num].name:
                continue
            if pointer == 1:
                context = (context + 
                template_Layers_WeightsBias.replace("[[number_layers]]",str(pointer))
                .replace("[[num_neurons_layer2]]",str(len(nnLayers[layer_num].get_weights()[1])-1))
                .replace("[[num_neurons_layer1]]",str(self.input_dim-1))
                .replace("[[DATA_TYPE]]", self.nn_data_type)
                )
            elif layer_num == len(nnLayers)-1:
                context = (context + 
                template_Output_WeightsBias
                .replace("[[num_neurons_layer2]]",str(self.output_dim-1))
                .replace("[[num_neurons_layer1]]",str(len(nnLayers[layer_num-1].get_weights()[1])-1))
                .replace("[[DATA_TYPE]]", self.nn_data_type)
                )
            else:
                context = (context + 
                template_Layers_WeightsBias.replace("[[number_layers]]",str(pointer))
                .replace("[[num_neurons_layer2]]",str(len(nnLayers[layer_num].get_weights()[1])-1))
                .replace("[[num_neurons_layer1]]",str(len(nnLayers[layer_num-1].get_weights()[1])-1))
                .replace("[[DATA_TYPE]]", self.nn_data_type)
                )
            pointer = pointer + 1

        if self.normalization:
            context = (context + 
                template_normalization_MeanStd
                .replace("[[num_neurons]]",str(self.input_dim-1))
                .replace("[[DATA_TYPE]]", self.nn_data_type)
                )
            
        if self.denormalization:
            context = (context + 
                template_denormalization_MeanStd
                .replace("[[num_neurons]]",str(self.output_dim-1))
                .replace("[[DATA_TYPE]]", self.nn_data_type)
                )

        return context

