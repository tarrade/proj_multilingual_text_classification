# Multilingual text classification using embedding, bert and deep learning

## Introduction

## Code structure

.  
├── LICENSE  
├── README.md  
├── data  
├── deployment  
│   ├── hp-tuning  
│   │   └── sklearn  
│   │       └── hyperparam.yaml  
│   └── training  
│       └── sklearn  
│           ├── custom.yaml  
│           └── standard.yaml  
├── doc  
│   ├── DOC.md  
│   ├── INSTRUCTION.md  
│   ├── SETUP.md  
│   └── img  
├── env  
│   └── environment.yml  
├── notebook  
│   ├── 00-Test  
│   │   ├── 01-Exploration-BigQuery.ipynb  
│   │   ├── 02-Visualization-facet-dive.ipynb  
│   │   ├── 02-Visualization-facet-overview.ipynb  
│   │   ├── 03-NLP-Preprocessing-Dataflow.ipynb  
│   │   ├── 03-Preprocessing-Dataflow.ipynb  
│   │   ├── 04-Data-selection.ipynb  
│   │   └── 05-Models.ipynb  
│   ├── 01-Exploration  
│   ├── 02-Preprocessing  
│   ├── 03-Models  
│   └── 04-Interpretation  
├── script  
├── src  
│   ├── analysis  
│   │   ├── __init__.py  
│   │   └── get_data.py  
│   ├── model  
│   │   └── sklearn_naive_bayes  
│   │       ├── __init__.py  
│   │       ├── model.py  
│   │       └── task.py  
│   ├── preprocessing  
│   │   ├── __init__.py  
│   │   └── preprocessing.py  
│   ├── setup.py  
│   └── utils  
│       ├── __init__.py  
│       ├── model_metrics.py  
│       ├── model_tests.py  
│       ├── model_utils.py  
│       └── ressources_utils.py  
└── test  

## General presentation and results of this project:
[Documentation](doc/DOC.md)

## Instruction to setup the project
[Configuration](doc/SETUP.md)

## How run the code
[Instruction](doc/INSTRUCTION.md)

## Results

## Conclusion