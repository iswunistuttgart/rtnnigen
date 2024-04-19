from typing import Tuple
import keras
from keras2plc.parse_model import keras_to_st_parser
from keras2plc.gen_st import ST_writer

def _is_all_dense_or_normalization(model)-> bool:
    all_layers_okay = True
    for i,layer in enumerate(model.layers):
        if i==0 or i==len(model.layers)-1:
            if not isinstance(layer, keras.layers.Dense) and not isinstance(layer, keras.src.layers.preprocessing.normalization.Normalization):
                layer_name = "First" if i==1 else "Last"
                print(f"{layer_name} layer is neither Dense nor Normalization layer.")
                all_layers_okay = False

        else: # all other layers:
            if not isinstance(layer, keras.layers.Dense):
                print(f"layer {i} is neither Dense nor Normalization layer.")
                all_layers_okay = False

    return all_layers_okay

def _get_io_dimensions(model: keras.Sequential) -> Tuple[int, int]:
    num_inputs = model.layers[0].input.shape[1]
    num_outputs = model.layers[-1].output.shape[1]
    return num_inputs, num_outputs

def keras2plc(keras_sequential_model: keras.Sequential, plc_model_name: str, plc_model_path: str, overwrite_if_model_exists: bool = False):
    if not _is_all_dense_or_normalization(keras_sequential_model):
        return
    
    n_inputs, n_outputs = _get_io_dimensions(keras_sequential_model)
    reader = keras_to_st_parser(keras_sequential_model, plc_model_name, input_dim=n_inputs, output_dim=n_outputs)
    layers_contents = reader.generate_struct_layers()
    layersWeights_contents = reader.generate_struct_layer_weights()

    writer = ST_writer(plc_model_name, layers_contents,layersWeights_contents)
    writer.write_ST_files_to(plc_model_path, overwrite_if_exists=overwrite_if_model_exists)

    weights_bin = reader.pack_weights_binary()
    writer.write_weights_file(weights_bin, overwrite_if_exists=overwrite_if_model_exists)

def get_example_usage(model : keras.Sequential, plc_model_name: str) -> str:
        """returns a string of example IEC 61131 code to call the generated model."""

        dims_input, dims_output = _get_io_dimensions(model)
        
        return f"""The following code can be used to call the generated model:
        Assuming declared input/output for model:
        
            input : ARRAY[0..{dims_input-1}] OF LREAL;
            result : ARRAY[0..{dims_output-1}] OF LREAL;

        Then call as:

            FB_{plc_model_name}(pointer_input:=ADR(input), pointer_output:=ADR(result));      
        
        """


