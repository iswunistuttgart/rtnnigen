# PYNN2PLC
## Description
This toolbox enables the generation of structured text from a keras neural network model. \
    ![use_case](/resources/diagram/use_case.png)


## How to use it:

### 1. Install the PLC library (shared code, mainly for matrix, vector operations)

- Open library manager: double-click on the References object in the PLC project tree\
    ![Library manager](/resources/pictures/library_manager.png)

- Open library Repository: click the button in Library Manager (symbol: Library Repository ![repository](/resources/pictures/repository.png))
- Install our library RTNNIgen: choose RTNNIgen.library after click install buttons\
    ![install](/resources/pictures/install.png)
- Add the necessary libraries: click button in Library Manager (symbol: add library ![add_library](/resources/pictures/add_library.PNG))\
    ![necessary_libraries](/resources/pictures/necessary_libraries.PNG)

### 2. Install RTNNIgen package in python:

In Python3 the package needs to be installed as

```sh
pip install -e .

# or directly from repo:
pip install git+https://git.isw.uni-stuttgart.de/projekte/forschung/2022_icm_nwg-gm/studentische-arbeiten/keras2plc.git@main
```

### 3. Convert any Keras Dense model in Python as follows:

```py
from keras2plc import keras2plc
import keras 

# load saved neural network model
model_file = "test_model.keras"
model = keras.saving.load_model(model_file) 
# (this step could be done differently, which depends on your tensorflow version

# where to save the structured text file and its name
folder = "ST_files/"
model_name = "Dense_v1"

# generate structured text
keras2plc(model, model_name, folder, overwrite_if_model_exists=True)
```
then several files have been generated in folder ST_files:

![generated_files](/resources/pictures/generated_files.png) 
### 4. Import the neural network code in TwinCAT
- add data types DUT (xx_Layers.TcDUT & xx_LayersWeights.TcDUT)
- add function block POU (FB_xx.TcPOU)

### 5. Example on the usage of generated code in TwinCAT
```py
from keras2plc import get_example_usage

print(get_example_usage(model, model_name))
```
Output :
```
The following code can be used to call the generated model:
    Assuming declared input/output for model:
    
        input : ARRAY[0..0] OF LREAL;
        result : ARRAY[0..0] OF LREAL;

    Then call as:

        FB_Dense_v1(pointer_input:=ADR(input), pointer_output:=ADR(result));  
```
## Project status

```
Date: 04. April  2024
1. Support only dense neural network
2. Limited usage of activation function: relu & tanh
```

