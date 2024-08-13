# DGL tutorial for ML4Chemistry beginners

## Requirements
Create a virtual environment to run the code to run DGL.<br>
Install pytorch with the cuda version that fits your device.<br>
```
cd dgl_tutorial
conda create -c conda-forge -n rdenv python=3.7 -y
conda activate rdenv
conda install pytorch -c pytorch -y
conda install -c conda-forge rdkit -y
pip install dgl
pip install dgllife
```

## Lessons

### Lesson 1. Molecule to DGL Graph
- Learn how to use rdkit and dgl to make a dgl graph from SMILES.
- Understand what is **features** in GNN models.

### Lesson 2. Building ML Model
- Making a very first MPNN model from scrath using dfl-life package.
- Understand how the features are updated inside the model.

### Lesson 3. Making DataLoader
- Making pytorch DataLoader to load the data for training and testing
- Understand how the data is preprocessed in a large scale

### Lesson 4. Train and Test
- Implement the real training and testing tasks for a given molecule property prediction task (odor classification).
- Understand the hyperparameters and the importance of each component in training a GNN model.

### Lesson 5. Customize it!
- Find your own proect and implement the models to learn and predict!