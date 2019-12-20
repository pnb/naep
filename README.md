# Nation's Report Card data mining competition entry (NAEP)

I used this competition as an opportunity to try out some new methods for educational data mining, especially focusing on automated feature engineering. The basic approach I settled on uses feature-level fusion from a few methods, so the process of generating the final predictions consists of generating individual feature sets, selecting (filtering) the features, and then fitting an ensemble machine learning model.

There are three basic types of features:

1. Ad-hoc features ("feature engineering") developed by reading log files and brainstorming
2. Automatically-engineered features using _TSFRESH_ (<https://github.com/blue-yonder/tsfresh>)
3. Automatically-engineered features using _FeatureTools_ (<https://github.com/FeatureLabs/featuretools>)

Note that it proved exceedingly difficult to exactly replicate the exact generated features in a new conda environment, even after keeping track of Python package numbers carefully (perhaps not as carefully as I thought). If you need the exact features generated to replicate the final predictions, contact me.

## Requirements

This has only been tested with Python 3.7, and may not work with other versions of Python.

I included the output of `pip freeze`, which can be used to recreate my Python environment. Create a conda environment with Python 3.7, and install the required packages:

    conda create --name naep_pnb python=3.7
    conda activate naep_pn
    pip install -r pip-freeze.txt

After installing, there are two modifications needed to fix bugs in TSFRESH and Scikit-Optimize.

### Patching TSFRESH

This bug will probably eventually be fixed, since TSFRESH is still under active development. However, to ensure the same (or as similar as possible) results, it would be good to use this specific fix:

1. Download <https://github.com/pnb/tsfresh/blob/45f5c63a38812c9de26977165bcb690d5ac782cf/tsfresh/feature_selection/benjamini_hochberg_test.py> (click "Raw" and Cmd+S or Ctrl+S to save the file)

2. Copy the file to the TSFRESH installation location in your Python environment (e.g., `~/anaconda3/envs/naep_pnb/lib/python3.7/site-packages/tsfresh/feature_selection/`)

### Patching Scikit-Optimize

Scikit-Optimize (skopt) is not currently under active development, so this fix may never make it into a new release.

1. Download <https://github.com/darenr/scikit-optimize/blob/180d6be130dfbead930093eef144d6dad171cfe5/skopt/searchcv.py>

2. Copy the file to the Scikit-Optimize installation location (e.g., `~/anaconda3/envs/naep_pnb/lib/python3.7/site-packages/skopt/`)

### Including the NAEP data

The NAEP data files (and features derived from them) are not included in the software repository to avoid data sharing concerns. The data files go in the `public_data` folder. You will need to add six CSV files to this folder:

    data_a_hidden_10.csv
    data_a_hidden_20.csv
    data_a_hidden_30.csv
    data_a_train.csv
    data_train_label.csv
    hidden_label.csv

## Feature extraction

### Timeseries features (TSFRESH)

Generate these features by running:

    python3 features_tsfresh.py
    python3 filter_features.py features_tsfresh

This should result in nine new CSV files in the `features_tsfresh/` folder, six of which are the actual feature data and three of which are the names of selected features.

### FeatureTools features

Run:

    python3 features_featuretools.py
    python3 filter_features.py features_featuretools

Like with TSFRESH, this should result in a total of nine new CSV files in the `features_featuretools` folder. This is usually a somewhat slow process (slightly over 1 hour).

### Ad-hoc features (traditional feature engineering)

Run:

    python3 features_fe.py
    python3 filter_features.py features_fe

## Model building

Run `python3 feature_level_fusion.py extratrees` to generate the predictions in the `predictions/` folder. This takes a few hours. Note that there are a few other options for this script, such as building XGBoost models or changing the hyperparameter optimization objective, but ultimately Extra-Trees seemed to work the best.

Run `python3 combine_predictions.py predictions/extratrees.csv` to post-process predictions. This step will do decision threshold adjustments and format the predictions appropriately for final submission in the `combine_predictions.txt` file. Optionally, it can also do decision-level fusion, but ultimately that did not prove helpful.

## Miscellaneous scripts (not needed, but interesting)

* `one_feature_exploration.py`: This script builds single-feature models with a single decision tree to provide a quick estimate of how good each individual feature is, independent of the others. It also builds models to see how good features are for predicting train vs. holdout, which is useful for identifying features that are unlikely to generalize.
* `ccp_alpha_explore.py`: This builds an Extra-Trees model with 500 trees and default hyperparameters, then finds the cost-complexity pruning (CCP) paths of each tree and graphs the alpha values that would result from pruning at each possible point. This is necessary to find the range of reasonable values for the CCP hyperparameter during model building. The range turns out to be about [0, .004] in this case.
* `features_similarity`: Features calculated from the Levenshtein distance between sequences. After calculating the distance matrix, I applied multidimensional scaling (MDS) and used the coordinates on the low-dimensional manifold as features. This seemed to work really well in cross-validation (see `one_feature_exploration.py` output), but did not generalize well to the leaderboard, so I removed these features from the final model.
* `naep_fusion_experiments.py`: An abandoned experiment in which I came up with a method to combine the predictions of two models, one with high AUC and one with high kappa, such that kappa would exactly preserved and AUC would be as close as possible to the better-AUC model. Ultimately this wasn't helpful since the best kappa model also had roughly the best AUC, but it is an interesting idea.
* `explore_data.py`: Lots of data exploration experiments that were helpful during ad-hoc feature engineering.
