# Altegrad-Protein-Prediction

Jérémie Dentan, Meryem Jaaidan, Abdellah El Mrini

This git repository implements some deep learning methods for protein classification. This classification task is part of a Kaggle challenge. For more details about the challenge, the data, and our approach, please refer to the `/doc` folder.

## Set up your environment

### Download the data

You should first download the data. To do so, please use the following link to download the data : [https://drive.google.com/file/d/1ybLD-EgKbeWVf6bKu796SYIF-1p-j7Nr/view?usp=share_link](https://drive.google.com/file/d/1ybLD-EgKbeWVf6bKu796SYIF-1p-j7Nr/view?usp=share_link).

Then, unzip the folder and copy its content directly in `data/`. The content of `data/` should now look like:

```bash
% ls -lah
total 4241392
drwxr-xr-x@ 10 jeremie  staff   320B  9 jan 15:42 .
drwxr-xr-x@ 11 jeremie  staff   352B  9 jan 18:29 ..
-rw-r--r--   1 jeremie  staff    13B  9 jan 18:29 .gitignore
-rw-rw-r--@  1 jeremie  staff   653M  7 déc 12:22 edge_attributes.txt
-rw-rw-r--@  1 jeremie  staff   211M  7 déc 12:21 edgelist.txt
-rw-rw-r--@  1 jeremie  staff   7,3M  7 déc 12:22 graph_indicator.txt
-rw-rw-r--@  1 jeremie  staff    42K  7 déc 17:13 graph_labels.txt
-rw-rw-r--@  1 jeremie  staff   1,2G  7 déc 12:22 node_attributes.txt
-rw-rw-r--@  1 jeremie  staff   1,5M  7 déc 15:56 sequences.txt
```

### Installing dependencies

This code is meant to run in **Python 3.8** with the PYTHONPATH set to the root of the project. We advise you to use [Python native virtual environments](https://docs.python.org/3/library/venv.html) or [Conda virtual environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

To do so, first install the base dependencies

```bash
pip install -r requirements.txt
```

Then, you need to install `torch-geometric`, `torch-scatter` and `torch-sparse`. The installation depends on your installation cuda, so we advise you to follow the official installation instructions:
* `torch-geometric`: [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
* `torch-sparse`: [https://pypi.org/project/torch-sparse/](https://pypi.org/project/torch-sparse/)
* `torch-scatter`: [https://pypi.org/project/torch-scatter/](https://pypi.org/project/torch-scatter/)

An easy way to do so is to set `CUDA` variable either to `cpu`, or `cu116` or `cu117` depending on your version of cuda, and the to get the wheels from Pypi. For cuda >11.7, e.g. CUDA 12.0, you can use `cu117` as well. For example:

```bash
export CUDA='cpu'
pip install torch-geometric==2.2.0
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
```

### Precomputed embeddings

Our prediction relies on embeddings of each proteins. This computation is really long, so we provide precomputed embeddings. They are available using the following link: [https://www.icloud.com/iclouddrive/058R2ZYAvPXvLt1IKELaqty_g#precomputedembeddings](https://www.icloud.com/iclouddrive/058R2ZYAvPXvLt1IKELaqty_g#precomputedembeddings)

Then, you should unzip and move the `embeddings` folder in `/output`, and it will be automatically recognized by our pipeline.

## Run the code

You first need to set up PYTHONPATH to the root of the project. Do do so, execute the following from the root:

```bash
export PYTHONPATH=$(pwd)
```

### Computing the embeddings

To compute the embeddings, you should run the following:

```bash
python -m src.embeddings
```

The embeddings will be automatically computed and stored in `/output/embeddings`. The logs of the computation will be stored in `/logs`.

Please note that this phase can be really long, depending on your hardware. If you use GPU, you will need at least 20Go of graphic memory. Using NVIDIA GeForce RTX 3090, the computation took:

* About 1h30 for the embeddings with `protbert`
* Few minutes for the embeddings with `tfidf`
* About 20min for the embeddings with `structure`

### Computing the predictions

To compute the predicted probabilities of each class, please run the following:

```bash
python -m src.predict
```

The output will be stored in `/output/submissions` and the logs in `/logs`. In particular, the log-loss on the validation set will be logged in `stdout` and stored in the logs.
