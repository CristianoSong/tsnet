# TSNet

This repository contains a reimplementation **TSNet**, a transformer-based foundation model for time-series forecasting and other analysis tasks. 
The model is designed for generalizable forecasting across heterogeneous time-series data, particularly in real-world applications with data distribution shifts and limited historical training data.


## Repository Structure

```bash
TSNet/
├── checkpoints/                # Directory for saving model checkpoints
├── data/                       # Directory for storing datasets
├── docs/                       # Documentation files
├── notebooks/                  # Jupyter notebooks for experiments and demos
│   ├── tsnet_freq_demo.ipynb   # Frequency attention demo notebook
├── tsnet/                      # Main source code for the TSNet project
│   ├── __init__.py             # Package initialization
│   ├── data_provider/          # Dataset processing and loading 
│   ├── layers/                 # Implementation of model layers
│   │   ├── attention.py        # Frequency attention mechanism
│   │   ├── decomposition.py    # Multi-series decomposition layer
│   │   ├── embedding.py        # Data embedding layer
│   │   ├── encoder_decoder.py  # Encoder and decoder layer
│   │   ├── projection_head.py  # Projector head for model output
│   ├── models/                 # Model definitions
│   │   ├── tsnet_freq.py       # TSNet model with frequency attention
│   │   ├── tsnet_patch.py      # TSNet model with patch embedding
│   ├── utils/                  # Utility functions
├── readme.md                   # Project README file
```

## Background and Disclaimer
This reimplementation is based on the TSNet architecture proposed in my research on time-series foundation modelling for intelligent communication network management systems. 
The original codebase was developed at Huawei Technologies and remains under internal confidentiality policies.
I claim that no proprietary code or internal data is included in this repository. 
All components and codes here are rebuilt from scratch, using public data and personal technical knowledge, after my departure from the company.
The current version is a prototype-level reimplementation for academic demonstration and discussion.
It may contain bugs or incomplete components, and will be continuously updated in future commits. 
Contributions and suggestions are very welcome.  