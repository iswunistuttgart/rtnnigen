import uuid
import os
from pathlib import Path
import logging

from nnigen.template_strings import (
    template_st_function_block_xml,
    template_st_struct_xml,
    template_st_struct,
    template_fb_inference_impl,
    template_fb_inference_decl,
)


class ST_writer:
    def __init__(self, unique_model_name: str, struct_layers_contents: str, struct_layersWeights_contents: str):
        self.model_name = unique_model_name
        self.nn_data_type = "LREAL"
        self.struct_contents = struct_layers_contents
        self.structWeights_contents = struct_layersWeights_contents
        self.path = "."

    def write_ST_files_to(self, path: str, overwrite_if_exists: bool = False):
        self.to_write = {}
        self.path = path
        self._add_fb_inference_file()
        self._add_nn_struct_file()
        self._add_nn_weights_struct_file()

        Path(path).mkdir(parents=True, exist_ok=True)

        # go through dict and write files
        for file_name, contents in self.to_write.items():
            file_path = os.path.join(path, file_name)
            if not overwrite_if_exists and os.path.exists(file_path):
                logging.warning(
                    f"File '{file_path}' exists and `write_ST_files_to` was not set to overwrite the old contents."
                    + "The existing model was not overwritten. Either rename the model or allow overwriting."
                )
            else:
                with open(file_path, "w") as f:
                    f.write(contents)

    def write_weights_file(self, bin_weights: bytes, overwrite_if_exists: bool = False):
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

            FB_{self.model_name}(pointer_input:=ADR(input), pointer_output:=ADR(result), nn:=GVL_{self.model_name}.nn);      
        
        """
    
    def _add_fb_inference_file(self):
        decl_part = self._get_fb_inference_decl()
        impl_part = self._get_fb_inference_impl()

        file_name = f"FB_{self.model_name}.st"
        file_contents = decl_part + "\n\n" + impl_part

        self.to_write[file_name] = file_contents

    def _add_nn_struct_file(self):
        uuid = TwinCAT_ST_writer.generate_uuid()
        st_struct_contents = self._get_st_struct_contents(
            struct_name=self._get_layers_struct_name(), struct_contents=self.struct_contents
        )

        file_name = f"{self._get_layers_struct_name()}.st"

        self.to_write[file_name] = st_struct_contents

    def _add_nn_weights_struct_file(self):
        uuid = TwinCAT_ST_writer.generate_uuid()
        st_weigths_struct_contents = self._get_st_struct_contents(
            struct_name=self._get_layersweights_struct_name(), struct_contents=self.structWeights_contents
        )

        file_name = f"{self._get_layersweights_struct_name()}.st"

        self.to_write[file_name] = st_weigths_struct_contents

    def _get_layers_struct_name(self) -> str:
        return self.model_name + "_Layers"

    def _get_layersweights_struct_name(self) -> str:
        return self.model_name + "_LayerWeights"

    def _get_fb_inference_decl(self) -> str:
        return (
            template_fb_inference_decl.replace("[[NAME]]", self.model_name)
            .replace("[[DATA_TYPE]]", self.nn_data_type)
            .replace("[[NAME_ST_LAYERS]]", self._get_layers_struct_name())
            .replace("[[WEIGHTS_FILE_PATH]]", self._get_layer_weights_path())
        )

    def _get_fb_inference_impl(self) -> str:
        return template_fb_inference_impl.replace("[[DATA_TYPE]]", self.nn_data_type)

    def _get_st_struct_contents(self, struct_name: str, struct_contents: str) -> str:
        return template_st_struct.replace("[[STRUCT_NAME]]", struct_name).replace(
            "[[STRUCT_CONTENTS]]", struct_contents
        )

    def _get_layer_weights_path(self) -> str:
        return os.path.abspath(os.path.join(self.path, f"{self.model_name}_weights.dat"))


class TwinCAT_ST_writer(ST_writer):
    def __init__(
        self,
        unique_model_name: str,
        struct_layers_contents: str,
        struct_layersWeights_contents: str,
        twincat_version: str = "3.1.4024.12",
    ):

        self.twincat_version = twincat_version
        super(TwinCAT_ST_writer, self).__init__(
            unique_model_name, struct_layers_contents, struct_layersWeights_contents
        )

    @classmethod
    def generate_uuid(cls) -> str:
        """returns a random uuid for the ST implementation."""
        return str(uuid.uuid4())


    def _add_fb_inference_file(self):
        uuid = TwinCAT_ST_writer.generate_uuid()
        decl_part = self._get_fb_inference_decl()
        impl_part = self._get_fb_inference_impl()

        file_name = f"FB_{self.model_name}.TcPOU"
        file_contents = (
            template_st_function_block_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[UUID]]", uuid)
            .replace("[[FB_DECL]]", decl_part)
            .replace("[[FB_IMPL]]", impl_part)
            .replace("[[NAME]]", self.model_name)
        )

        self.to_write[file_name] = file_contents

    def _add_nn_struct_file(self):
        uuid = TwinCAT_ST_writer.generate_uuid()
        st_struct_contents = self._get_st_struct_contents(
            struct_name=self._get_layers_struct_name(), struct_contents=self.struct_contents
        )
        file_name = f"{self._get_layers_struct_name()}.TcDUT"
        file_contents = (
            template_st_struct_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[NAME_ST_LAYERS]]", self._get_layers_struct_name())
            .replace("[[STRUCT_DEF]]", st_struct_contents)
            .replace("[[UUID]]", uuid)
        )

        self.to_write[file_name] = file_contents

    def _add_nn_weights_struct_file(self):
        uuid = TwinCAT_ST_writer.generate_uuid()
        st_weigths_struct_contents = self._get_st_struct_contents(
            struct_name=self._get_layersweights_struct_name(), struct_contents=self.structWeights_contents
        )

        file_name = f"{self._get_layersweights_struct_name()}.TcDUT"
        file_contents = (
            template_st_struct_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[NAME_ST_LAYERS]]", self._get_layers_struct_name())
            .replace("[[STRUCT_DEF]]", st_weigths_struct_contents)
            .replace("[[UUID]]", uuid)
        )

        self.to_write[file_name] = file_contents
