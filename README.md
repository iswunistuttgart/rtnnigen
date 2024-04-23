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

### 2. Python side:

In Python3 the package needs to be installed as

```sh
pip install -e .

# or directly from repo:
pip install git+https://git.isw.uni-stuttgart.de/projekte/forschung/2022_icm_nwg-gm/studentische-arbeiten/keras2plc.git@main
```

### 3. Convert any Keras Dense model in Python as follows:

```py
# TODO: get twincat version from ADS

# TODO: Add usage examples
```

### 4. Import the neural network code in TwinCAT


#### 5. Send the weights

```py
# TODO: add python code to send weights (check for correct model name and dimensions!)
```


## Project status
```
Date: 04. April  2024
1. Support only dense neural network
2. Limited usage of activation function: relu & tanh
```

