from typing import Tuple
import keras
from nnigen.parse_model import keras_to_st_parser
from nnigen.gen_st import TwinCAT_ST_writer, ST_writer


def nnigen(
    keras_sequential_model: keras.Sequential,
    plc_model_name: str,
    plc_model_path: str,
    overwrite_if_model_exists: bool = False,
    write_plain_st: bool = False,
):
    """ converts a given `keras.Sequential` model to TwinCAT ST files.
     
    ### Inputs:
    
    keras_sequential_model                          ... the Keras model to generate a PLC model from
    plc_model_name: str                             ... a unique model name to distinguish the model from others in the PLC
    plc_model_path : str                            ... the path to export the model to. If nonexistent, the path will be generated.
    overwrite_if_model_exists: bool [default: False] ... Flag, whether to oveewrite files, if model files exist already.
    write_plain_st: bool [default: False]           ... 

    ### Outputs:

    written to files directly
      """
    reader = keras_to_st_parser(keras_sequential_model, plc_model_name)
    #layers_contents = reader.generate_struct_layers()
    #layersWeights_contents = reader.generate_struct_layer_weights()

    if write_plain_st:
        writer = ST_writer(plc_model_name, reader)
    else:  # output TwinCAT3 ready xml files
        writer = TwinCAT_ST_writer(plc_model_name, reader)

    writer.write_ST_files_to(plc_model_path, overwrite_if_exists=overwrite_if_model_exists)
    writer.write_weights_file( overwrite_if_exists=overwrite_if_model_exists)


def get_example_usage(keras_sequential_model: keras.Sequential, plc_model_name: str) -> str:
    """returns a string of example IEC 61131 code to call the generated model."""
    reader = keras_to_st_parser(keras_sequential_model, plc_model_name)
    dims_input, dims_output = reader._get_io_dimensions()

    return f"""The following code can be used to call the generated model:
        Assuming declared input/output for model:
        
            input : ARRAY[0..{dims_input-1}] OF LREAL;
            result : ARRAY[0..{dims_output-1}] OF LREAL;

        Then call as:

            FB_{plc_model_name}(pointer_input:=ADR(input), pointer_output:=ADR(result));      
        
        """
