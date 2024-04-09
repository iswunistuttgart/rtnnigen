import keras

def keras2plc(keras_sequential_model, plc_numeric_data_type:str="LREAL") -> (str, str, str):
    def is_all_dense_or_normalization(model)-> bool:
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



    model = keras_sequential_model

    if not is_all_dense_or_normalization(model):
        return

    input_length = model.input_shape[1]
    num_neurons = [layer.units for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    num_dense_layers = len(num_neurons)
    has_input_normalization = isinstance(model.layers[0], keras.src.layers.preprocessing.normalization.Normalization)
    has_output_normalization = isinstance(model.layers[-1], keras.src.layers.preprocessing.normalization.Normalization)

    text_layers_allocation = "\n".join([f"\tHiddenLayer{i+1}_output : ARRAY[0..{num_neurons_i-1}] OF LREAL;" for i,num_neurons_i in enumerate(num_neurons)])
    text_layers_counter_indices = "\n".join([f"\ti{i} : INT := 0;" for i in range(len(num_neurons))])

    text_decl = f"""
FUNCTION_BLOCK FB_NN_Forward_Propagation
// calculation the forward propagation of neural network
// include normalization layer and denormalization layer
VAR_INPUT
	// input : [x, tau, v] at time k
	input : ARRAY[0..{input_length-1}] OF LREAL;
END_VAR
VAR_OUTPUT
	// output [delta_v] at time k+1
	output : LREAL;
END_VAR
VAR
	// activation function
    activation : FB_Relu;
	// pre-allocation of layer outputs
{text_layers_allocation}

    // input normalizer
	normalizer : FB_Normalization_Input;
	// output denormalizer
	denormalizer : FB_Denormalization_Output;
	// normalized input
	input_norm : ARRAY[0..{input_length-1}] OF LREAL;	
	// normalized output
	output_norm : LREAL;
	// counters
{text_layers_counter_indices}

END_VAR
"""
    
    text_impl = ""
    
    return text_decl, text_impl