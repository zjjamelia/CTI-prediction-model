CTI Prediction Model

Code for the inflammation-related compound–target interaction (CTI) prediction model described in Deep Learning-Driven Identification and Experimental Validation of COX-2 and ERK1/2 as Anti-Inflammatory Targets of Avenanthramides.

Contents
- `python/`: model scripts (DAE, CNN training and prediction)
- `positive_pairs.csv`: positive CTI pairs (DrugBank IDs and UniProt IDs)

Data availability
The original DrugBank data are not redistributed due to license restrictions. 
Positive CTI pairs are provided in positive_pairs.csv as DrugBank IDs and UniProt IDs.

Requirements
Python 3.x, PyTorch, scikit-learn, numpy, pandas

License
Apache License 2.0