# RTNNIgen: Code generation from Keras for sequential neural networks to run them directly in the PLC (TwinCAT3)

![rtnnigen logo](/resources/pictures/rtnnigen_logo_256.png)

![GitHub License](https://img.shields.io/github/license/iswunistuttgart/rtnnigen)


This toolbox enables the generation of TwinCAT3 Structured Text from a Keras `sequential` neural network model. It is driven by two components:

- `RTNNI`: a PLC library allowing real-time capable neural network inference. This avoids redundancies in the code generation step as well as providing a cleaner (more readable) interface
- `nnigen`: a Python package converting Keras sequential models to PLC code using the `RTNNI` library.


The following use-cases are supported:

![use_case](/resources/diagram/use_case.png)

## Contents

- [RTNNIgen: Code generation from Keras for sequential neural networks to run them directly in the PLC (TwinCAT3)](#rtnnigen-code-generation-from-keras-for-sequential-neural-networks-to-run-them-directly-in-the-plc-twincat3)
  - [Contents](#contents)
  - [How to install](#how-to-install)
    - [PLC library `RTNNI`](#plc-library-rtnni)
    - [`nnigen` package in Python:](#nnigen-package-in-python)
  - [How to use](#how-to-use)
    - [Code generation for Keras `sequential` model in Python](#code-generation-for-keras-sequential-model-in-python)
    - [Import the generated code in the TwinCAT project](#import-the-generated-code-in-the-twincat-project)
    - [Generate a usage example in Python for TwinCAT](#generate-a-usage-example-in-python-for-twincat)
    - [Update weights only (e.g. after retraining)](#update-weights-only-eg-after-retraining)



## How to install

### PLC library `RTNNI`

- Open the Library Manager: double-click on the References object in the PLC project tree\
    ![Library Manager](/resources/pictures/library_manager.png)

- Open the Library Repository ![Repository](/resources/pictures/repository.png): click the button in Library Manager
- Install our library RTNNI: click on the `Install...` button (shown below), then choose `RTNNI.library`

    ![install](/resources/pictures/install.png)

- Add the necessary libraries: click the button  add library ![add_library](/resources/pictures/add_library.PNG) in Library Manager and add the following dependencies:
  
    ![necessary_libraries](/resources/pictures/necessary_libraries.PNG)

### `nnigen` package in Python:

In Python3 the package needs to be installed. This can be done via

```sh
# install from cloned repository 
git clone https://github.com/iswunistuttgart/rtnnigen.git
cd rtnnigen
pip install .

# or directly from repo:
pip install git+https://github.com/iswunistuttgart/rtnnigen.git@main
```

## How to use

### Code generation for Keras `sequential` model in Python

```py
from nnigen import nnigen
import keras 

# load saved neural network model
model_file = "test_model.keras"
model = keras.saving.load_model(model_file) 
# (this step could be done differently, which depends on your tensorflow version

# tip: directly use a subfolder in the PLC project, nonexistent folders will be created
folder = "ST_files/"
model_name = "Dense_v1"

# generate structured text
nnigen(model, model_name, folder, overwrite_if_model_exists=True)
```
then several files have been generated in folder ST_files:

| File | Contents |
|--|--|
| `{model_name}_LayersWeights.TcDUT` | Struct containing all model weights (the variable part of the model) |
| `{model_name}_weights.dat` | Binary serialized weights (corresponding to `{model_name}_LayersWeights.TcDUT`)
| `{model_name}_Layers.TcDUT` | Struct containing the whole network |
| `FB_{model_name}.TcPOU` | Function block for model inference (forward pass). Loads the weights on initialization (first >6 calls). This is the only component of the model that needs to be accessed. |

For the code example above, the generated set of files would be:

![generated_files](/resources/pictures/generated_files.png) 


### Import the generated code in the TwinCAT project

Add data types DUT (**{model_name}_Layers.TcDUT** & **{model_name}_LayersWeights.TcDUT**) and function block POU (**FB_{model_name}.TcPOU**)
 
right click on your PLC project -> choose "add" -> choose "existing item..." -> choose files

    ![add_function](/resources/pictures/add_function.PNG) 

> **Warning:** The path of the model weights is coded into variable `filePath` of `FB_{model_name}.TcPOU`. If you move the weights file after its creation make sure to adapt the path in `filePath`. Otherwise, loading the weights will fail.

### Generate a usage example in Python for TwinCAT

```py
from nnigen import get_example_usage

print(get_example_usage(model, model_name))
```

Example output :

```
The following code can be used to call the generated model:
    Assuming declared input/output for model:
    
        input : ARRAY[0..0] OF LREAL;
        result : ARRAY[0..0] OF LREAL;

    Then call as:

        FB_Dense_v1(pointer_input:=ADR(input), pointer_output:=ADR(result));  
```

### Update weights only (e.g. after retraining)

In Python 

```py
from nnigen import update_model_weigths

# assuming `model` is a Keras sequential model
# which was previously exported and now its weights were retrained


folder = "ST_files/"
model_name = "Dense_v1"

update_model_weigths(model, model_name, folder)
```

> **Warning:** If the export location of the weights differs from the folder used for the original export, also adapt the variable `filePath`of `FB_{model_name}.TcPOU` to let the PLC know the new weights location.