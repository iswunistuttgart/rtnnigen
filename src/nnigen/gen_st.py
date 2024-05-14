import uuid
import os
from pathlib import Path
import logging

from nnigen.parse_model import clean_indentation, model_parser
from nnigen.template_strings import (
    template_st_function_block_xml,
    template_st_struct_xml,
    template_st_struct,
    template_fb_inference_impl,
    template_fb_inference_decl,
)


class ST_writer:
    """basic class to write ST contents"""

    to_write = {}
    """ dictionary being written to the file system. keys (str) are file names, values are file contents"""

    def __init__(self, unique_model_name: str, parser: model_parser):
        """ST_writer __init__

        ### Inputs:

        unique_model_name: str ... name of the model (used in file names to distinguish models)
        parser: `model_parser` subclass to generate structured text
        """
        self.model_name = unique_model_name
        self.nn_data_type = "LREAL"
        self.parser = parser
        self.path = "."

    def write_ST_files_to(self, path: str, overwrite_if_exists: bool = False):
        """writes the ST files

        ### Inputs:

        path : str                                ... path to export the files to (can be in the PLC project).
                                                      Will be created if nonexistent.
        overwrite_if_exists: bool [default:False] ... if the model was previously exported (files already exist),
                                                       this flag defines whether they can be overwritten. Otherwise,
                                                       a warning is given to the user.
        """
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

    def write_weights_file(self, overwrite_if_exists: bool = False):
        """
        Save weights and bias of all layers into a binary file, which can be
        loaded automatically when neural network is initialized in PLC

        ### Inputs:

        overwrite_if_exists: bool [default:False] ... if the model was previously exported (files already exist),
                                                       this flag defines whether they can be overwritten. Otherwise,
                                                       a warning is given to the user.
                                             
        """

        # write file
        file_path = self._get_layer_weights_path()

        if not overwrite_if_exists and os.path.exists(file_path):
            logging.warning(
                f"File '{file_path}' exists and `generate_weights_file` was not set to overwrite the old contents."
                + "The existing model was not overwritten. Either rename the model of allow overwriting."
            )
        else:
            with open(file_path, "wb") as f:
                f.write(self.parser.pack_weights_binary())

    def _add_fb_inference_file(self):
        """internal function to query the function block for model inference for writing."""
        decl_part = self._get_fb_inference_decl()
        impl_part = self._get_fb_inference_impl()

        file_name = f"FB_{self.model_name}.st"
        file_contents = decl_part + "\n\n" + impl_part

        self.to_write[file_name] = file_contents

    def _add_nn_struct_file(self):
        """internal function to query the neural network data structure for writing."""
        uuid = TwinCAT_ST_writer.generate_uuid()
        st_struct_contents = self._get_st_struct_contents(
            struct_name=self._get_layers_struct_name(), struct_contents=self.parser.generate_struct_layers()
        )

        file_name = f"{self._get_layers_struct_name()}.st"

        self.to_write[file_name] = st_struct_contents

    def _add_nn_weights_struct_file(self):
        """internal function to query the neural network weights structure for writing."""
        uuid = TwinCAT_ST_writer.generate_uuid()
        st_weigths_struct_contents = self._get_st_struct_contents(
            struct_name=self._get_layersweights_struct_name(),
            struct_contents=self.parser.generate_struct_layer_weights(),
        )

        file_name = f"{self._get_layersweights_struct_name()}.st"

        self.to_write[file_name] = st_weigths_struct_contents

    def _get_layers_struct_name(self) -> str:
        return self.model_name + "_Layers"

    def _get_layersweights_struct_name(self) -> str:
        return self.model_name + "_LayerWeights"

    def _get_fb_inference_decl(self) -> str:
        """return the declaration part of the inference function block"""
        return (
            template_fb_inference_decl.replace("[[NAME]]", self.model_name)
            .replace("[[DATA_TYPE]]", self.nn_data_type)
            .replace("[[NAME_ST_LAYERS]]", self._get_layers_struct_name())
            .replace("[[WEIGHTS_FILE_PATH]]", self._get_layer_weights_path())
        )

    def _get_fb_inference_impl(self) -> str:
        """return the implementation part of the inference function block"""

        if self.parser.has_normalization:
            norm_impl = clean_indentation(
                """// input normalization
                            F_NormalizationLayer(pointer_input := ADR(nn.layer_input),pointer_mean := ADR(nn.weights.normalization_mean),
                                pointer_std := ADR(nn.weights.normalization_std),invert := FALSE, num_neurons := nn.layers[0].num_neurons); """,
                " " * 4,
            )
        else:
            norm_impl = ""

        if self.parser.has_denormalization:
            denorm_impl = clean_indentation(
                """// output denormalization
                            F_NormalizationLayer(pointer_input := pointer_output,pointer_mean := ADR(nn.weights.denormalization_mean),
                                pointer_std := ADR(nn.weights.denormalization_std),invert := TRUE, num_neurons := nn.layers[SIZEOF(nn.layers)/SIZEOF(nn.layers[0])-1].num_neurons);""",
                " " * 4,
            )
        else:
            denorm_impl = ""

        return (
            template_fb_inference_impl.replace("[[DATA_TYPE]]", self.nn_data_type)
            .replace("[[NORMALIZATION]]", norm_impl)
            .replace("[[DENORMALIZATION]]", denorm_impl)
        )

    def _get_st_struct_contents(self, struct_name: str, struct_contents: str) -> str:
        """builds a ST struct with given `name` and contents (`struct_contents` as IEC61131-3 code)"""
        return template_st_struct.replace("[[STRUCT_NAME]]", struct_name).replace(
            "[[STRUCT_CONTENTS]]", struct_contents
        )

    def _get_layer_weights_path(self) -> str:
        """helper function to get the full absolute path of the serialized model weigths."""
        return os.path.abspath(os.path.join(self.path, f"{self.model_name}_weights.dat"))


class TwinCAT_ST_writer(ST_writer):
    """Subclass of `ST_writer` for writing TwinCAT XML files."""

    def __init__(
        self,
        unique_model_name: str,
        parser: model_parser,
        twincat_version: str = "3.1.4024.12",
    ):

        self.twincat_version = twincat_version
        super(TwinCAT_ST_writer, self).__init__(unique_model_name, parser)

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
            struct_name=self._get_layers_struct_name(), struct_contents=self.parser.generate_struct_layers()
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
            struct_name=self._get_layersweights_struct_name(),
            struct_contents=self.parser.generate_struct_layer_weights(),
        )

        file_name = f"{self._get_layersweights_struct_name()}.TcDUT"
        file_contents = (
            template_st_struct_xml.replace("[[TWINCAT_VERSION]]", self.twincat_version)
            .replace("[[NAME_ST_LAYERS]]", self._get_layersweights_struct_name())
            .replace("[[STRUCT_DEF]]", st_weigths_struct_contents)
            .replace("[[UUID]]", uuid)
        )

        self.to_write[file_name] = file_contents
