# pytorch-wildlife-test-run

This repo tests if AI4G models can be deployed using the EcoAssist workflow. In `classify.py` you can find a minimal reproducable example. Feel free to test by cloning the repo and PR with adjustments.

EcoAssist works with the following workflow to classify animals:
1. loop through the MegaDetector output JSON to find where the animals are
2. crop out each animal and feed it to the classifier
3. add its predictions to the existing JSON

There are currently two issues:
1. 

In order to run this test:
1. It is assumed you have a conda environment called `pytorch-wildlife`.
    ```
    conda create -n pytorch-wildlife python=3.8 -y
    conda activate pytorch-wildlife
    pip install PytorchWildlife
    ```
2. You have downloaded the model file from Zenodo and placed it in the repo folder: https://zenodo.org/records/10042023/files/AI4GAmazonClassification_v0.0.0.ckpt?download=1

Execute script:
```
conda activate pytorch-wildlife && python "pytorch-wildlife-test-run\classify.py"
```
