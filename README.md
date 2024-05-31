# Multicalibration Post-Processing

This repository contains the official implementation of all experiments in "Is Multicalibration Post-Processing Necessary?" We conduct the first
comprehensive empirical study of multicalibration post-processing,
across a broad set of tabular, image, and language datasets for models spanning
from simple decision trees to 90 million parameter fine-tuned LLMs.

Included in this repository is all code necessary to run experiments from the paper, but it may also serve as a standalone tool for studying multicalibration. 

## Requirements

Original experiments were run in a Conda environment using Python 3.11. Before proceeding, we recommend updating pip. To install necessary packages:

```setup
pip install -r requirements.txt
```

## Reproducing Results

All experimental results are provided in the `results` directory. To create figures from these results, run the following commands in the root directory:

```bash
python scripts/generate_figures.py
python scripts/generate_tables.py
```

To run an experiment, run one of the functions available in `experiments.py`. Given a model, dataset, list of calibration fractions, and list of seeds for the validation split, these functions will pretrain, train, or evalaute (depending on the function) over the specified calibration fractions and split seeds. To specify the model hyperparameters on each dataset and calibration fraction, one may edit the `hyperparameters` dictionary in `configs/hyperparameters.py`, though it currently contains the hyperparameters used to obtain our results.

Once models have been trained, post-processed, and evaluated, one may reproduce the figures on these new runs. To first download the results from wandb, run the script provided in `download_results.py`. This will download the entire collection of wandb runs as csvs, which will be stored in the `results` directory. Once this information is saved, one may freely generate figures with the scripts cited above.

## Using This Repository

Training a model and applying a post-processing algorithm is straightforward. Consider the following example, which retrieves hyperparameters we use in the paper.

First define an mcb algorithm. To see available algorithms, examine the names and parameter dictionaries in `configs/mcb_algorithms.py`. Alternatively, one may look at their implementations in the `mcb_algorithms` directory.

```python
mcb_algorithm = 'HKRR'
mcb_params = {
    'lambda': 0.1,
    'alpha': 0.025,
}
```

From here, running each postprocess follows from calling `experiment.multicalibrate()` with the desired algorithm and parameters. Due to the computational cost of these algorithms, it is sometimes easier to run multiple after training a single base predictor. To do this, one may use `experiment.multicalibrate_multiple()`; for usage instructions, see the docstring in `Experiment.py`.

```python
from configs.constants import SPLIT_DEFAULT, MCB_DEFAULT
from configs.hyperparameters import get_hyperparameters
from Experiment import Experiment
from Dataset import Dataset
from Model import Model

# set constants for the experiment
model_name = 'MLP'
mcb_algorithm = 'HKRR'
mcb_params = {'lambda': 0.1, 'alpha': 0.025}
dataset = 'ACSIncome'
calib_frac = 0.4
seed = 0

# set the save directory and wandb project
save_dir = f'models/saved_models/{dataset}/{model_name}/calib={calib_frac}_val_seed={seed}/'
wdb_project = f'{dataset}_project'

# define config for experiment
hyp = get_hyperparameters(model_name, dataset, calib_frac)
config = {
    'model': model_name,        # track model in log
    'mcb': [mcb_algorithm],     # track mcb algorithm in log
    'dataset': dataset,         # track dataset name in log
    'calib_frac': calib_frac,   # fraction of train set used in mcb
    'val_split_seed': seed,     # seed for validation split
    'split': SPLIT_DEFAULT,     # train-val-test split
    'save_dir': save_dir,       # where to save model, if appropriate
    'val_save_epoch': 0,        # save model when val_save_epoch >= 0
    'val_eval_epoch': 1,        # eval model when (epoch % val_eval_epoch) == 0
    **hyp
}

dataset_obj = Dataset(dataset, val_split_seed=config['val_split_seed'])
model = Model(model_name, config=config, SAVE_DIR=config['save_dir'])
experiment = Experiment(dataset_obj, model, calib_frac=config['calib_frac'])

# init logger; this saves metrics to wandb
experiment.init_logger(config, project=wdb_project)

# train and postprocess
experiment.train_model()
if config['calib_frac'] > 0:
    experiment.multicalibrate(mcb_algorithm, mcb_params)

# evaluate splits
experiment.evaluate_val()
experiment.evaluate_test()

# close logger
experiment.init_logger(finish=True)
```

While this example uses pre-defined hyperparameters for the base predictor, it is possible to specify custom hyperparameters. To do this, change the appropriate key-value pairs on the `config` dictionary. To see available and required hyperparameters for each base predictor, examine `configs/hyperparameters.py`.


## Acknowledgements

The [WILDS Benchmark](https://github.com/p-lambda/wilds) provided partial inspiration for the design of this repository. We thank Bhavya Vasudeva for sharing resources that became helpful in its development.
