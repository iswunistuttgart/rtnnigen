# PYNN2PLC
## Description
This is a toolbox which can convert the tensorflow neural network model into the Structured Text.

```mermaid
flowchart LR
    subgraph Twincat
    S[Structured Text]
    G[Global Varialble]
    end
    A[your_NN_model.keras] -->|generate| S & G
    A -->|get weights + bias| P[Pyads]
    P --> |send| G
    subgraph Python
    A & P
    end
```
## Project status
```
Date: 04. April  2024
1. Support only dense neural network
2. Limited usage of activation function: relu & tanh
```

