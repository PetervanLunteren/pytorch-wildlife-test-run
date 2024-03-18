# pytorch-wildlife-test-run

This repo tests if AI4G models can be deployed using the EcoAssist workflow
In `classify.py` you can find a minimal reproducable example
You test it yourself by cloning the repo
Feel free to PR with adjustments

EcoAssist works with the following workflow to classify animals:
1. loop through the MegaDetector output JSON to find where the animals are
2. crop out each animal and feed it to the classifier
3. add its predictions to the existing JSON

It is assumed you have a conda environment called `pytorch-wildlife`. Execute script:
```
conda activate pytorch-wildlife && python "pytorch-wildlife-test-run\classify.py"
```
