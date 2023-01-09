# Altegrad-Protein-Prediction

Jérémie Dentan, Meryem Jaaidan, Abdellah El Mrini

## Execute the code

### Download the data

You should first download the data. To do so, please use the following link to download the data : [https://drive.google.com/file/d/1ybLD-EgKbeWVf6bKu796SYIF-1p-j7Nr/view?usp=share_link](https://drive.google.com/file/d/1ybLD-EgKbeWVf6bKu796SYIF-1p-j7Nr/view?usp=share_link).

Then, unzip the folder and copy its content directly in `data/`. The content of `data/` should now look like:

```bash
% ls -lah
total 4241392
drwxr-xr-x@ 10 jeremie  staff   320B  9 jan 15:42 .
drwxr-xr-x@ 11 jeremie  staff   352B  9 jan 18:29 ..
-rw-r--r--@  1 jeremie  staff   6,0K  9 jan 15:42 .DS_Store
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

You can install the dependencies using:

```bash
pip install -r requirements.txt
```
