# DVC Workflow

## 1. Dataset Tracking

DVC is used to track and version the Iris dataset.

The dataset is added to DVC using:

dvc add data/raw/iris_v1.csv

This creates a `.dvc` metadata file while the actual dataset is stored in the DVC cache.

## 2. Git Integration

Git tracks the `.dvc` file, while DVC tracks the actual dataset.

For each dataset version:

git add data/raw/iris_v1.csv.dvc
git commit -m "data: update iris dataset"

## 3. DVC Remote

A local DVC remote is configured to store dataset versions.

The dataset is uploaded using:

dvc push

## 4. Dataset Versioning

Version 1 contains 150 rows.

Version 2 contains 170 rows after adding 20 synthetic rows.

DVC allows us to switch between these dataset versions.

## 5. Comparing Versions

Dataset versions can be compared using:

dvc diff <commit>

This shows which datasets have been modified.

## 6. Reproducing an Older Version

To reproduce an older dataset version:

git checkout <commit> -- data/raw/iris_v1.csv.dvc
dvc checkout data/raw/iris_v1.csv.dvc

This restores the dataset associated with that Git commit.

## 7. Workflow Summary

The complete workflow is:

1. Create the dataset.
2. Track the dataset using DVC.
3. Commit the `.dvc` file using Git.
4. Push dataset contents using `dvc push`.
5. Modify the dataset.
6. Create a new DVC version.
7. Compare versions using `dvc diff`.
8. Restore previous versions using Git and `dvc checkout`.

This workflow provides dataset versioning and reproducibility for machine learning projects.