import uuid
import os
from pathlib import Path
import logging

from nnigen.template_strings import template_st_function_block_xml, template_st_struct_xml


class ST_writer:
    def __init__(
        self,
        unique_model_name: str,
        struct_layers_contents: str,
        struct_layersWeights_contents: str,
        twincat_version: str = "3.1.4024.12",
    ):
        self.model_name = unique_model_name
        self.twincat_version = twincat_version
        self.nn_data_type = "LREAL"
        self.struct_contents = struct_layers_contents
        self.structWeights_contents = struct_layersWeights_contents
        self.path = "."

    def write_ST_files_to(self, path: str, overwrite_if_exists: bool = False):
        self.to_write = {}
        self.path = path
        self._add_fb_inference(path)
        self._add_nn_struct()
        self._add_nn_weights_struct()

        Path(path).mkdir(parents=True, exist_ok=True)

        for file_name, contents in self.to_write.items():
            file_path = os.path.join(path, file_name)
            if not overwrite_if_exists and os.path.exists(file_path):
                logging.warning(
                    f"File '{file_path}' exists and `write_ST_files_to` was not set to overwrite the old contents."
                    + "The existing model was not overwritten. Either rename the model of allow overwriting."
                )
            else:
                with open(file_path, "w") as f:
                    f.write(contents)

    @classmethod
    def generate_uuid(cls) -> str:
        """returns a random uuid for the ST implementation."""
        return str(uuid.uuid4())

    def _get_layers_struct_name(self) -> str:
        return self.model_name + "_Layers"
    
    def _get_layersweights_struct_name(self) -> str:
        return self.model_name + "_LayerWeights"
    
    def _get_layer_weights_path(self,path : str) -> str:
        return os.path.abspath(os.path.join(path,f'{self.model_name}_weights.dat'))

    def _add_fb_inference(self,path : str):
        uuid = ST_writer.generate_uuid()
        file_name = f"FB_{self.model_name}.TcPOU"
        file_contents = (
            template_st_function_block_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[NAME]]", self.model_name)
            .replace("[[DATA_TYPE]]", self.nn_data_type)
            .replace("[[UUID]]", uuid)
            .replace("[[NAME_ST_LAYERS]]", self._get_layers_struct_name())
            .replace("[[filePath_weights]]",self._get_layer_weights_path(path))
        )

        self.to_write[file_name] = file_contents

    def _add_nn_struct(self):
        uuid = ST_writer.generate_uuid()
        file_name = f"{self._get_layers_struct_name}.TcDUT"
        file_contents = (
            template_st_struct_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[NAME_ST_LAYERS]]", self._get_layers_struct_name())
            .replace("[[STRUCT_CONTENTS]]", self.struct_contents)
            .replace("[[UUID]]", uuid)
        )

        self.to_write[file_name] = file_contents

    def _add_nn_weights_struct(self):
        uuid = ST_writer.generate_uuid()
        file_name = f"{self._get_layersweights_struct_name()}.TcDUT"
        file_contents = (
            template_st_struct_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[NAME_ST_LAYERS]]", self._get_layersweights_struct_name())
            .replace("[[STRUCT_CONTENTS]]", self.structWeights_contents)
            .replace("[[UUID]]", uuid)
        )

        self.to_write[file_name] = file_contents

    def write_weights_file(self, bin_weights : bytes, overwrite_if_exists : bool = False):
        """
        Save weights and bias of all layers into a binary file, which can be 
        loaded automatically when neural network is initialized in PLC
        """

        
        # write file
        file_name = f"{self.model_name}_weights.dat"
        file_path = os.path.join(self.path, file_name)
        if not overwrite_if_exists and os.path.exists(file_path):
            logging.warning(
                f"File '{file_path}' exists and `generate_weights_file` was not set to overwrite the old contents."
                + "The existing model was not overwritten. Either rename the model of allow overwriting."
            )
        else:
            with open(file_path, "wb") as f:
                f.write(bin_weights)


    def create_example_usage(self, dims_input: int, dims_output: int) -> str:
        """returns a string of example IEC 61131 code to call the generated model."""
        return f"""The following code can be used to call the generated model:
        Assuming declared input/output for model:
        
            input : ARRAY[0..{dims_input-1}] OF {self.nn_data_type};
            result : ARRAY[0..{dims_output-1}] OF {self.nn_data_type};

        Then call as:

            FB_{self.model_name}(pointer_input:=ADR(input), pointer_output:=ADS(result), nn:=GVL_{self.model_name}.nn);      
        
        """