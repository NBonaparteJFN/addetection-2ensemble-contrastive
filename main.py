import dataset
import ensemble_trainer
import evaluator
import utils
import sys
import numpy as np

import pdb

from config import config


def main(config):
    # Create save directories
    utils.create_directories(config)

    # Prepare and load the data
    if 'silences' in config.model_types:
        data = dataset.prepare_data_new(config.dataset_dir, config)
    else:
        data = dataset.prepare_data(config.dataset_dir, config)
    # print(data)
    # return
    # Train the ensemble models
    if config.training_type == 'bagging':
        ensemble_trainer.bagging_ensemble_training(data, config)
    elif config.training_type == 'boosting':
        ensemble_trainer.boosted_ensemble_training(data, config)

    # Evaluate the model
    if 'silences' not in config.model_types:
        test_data = dataset.prepare_test_data(config.test_dataset_dir, config)
        evaluator.evaluate(data, test_data, config)


def mainEvaluate(config):
    data = dataset.prepare_data(config.dataset_dir, config)
    test_data = dataset.prepare_test_data(config.test_dataset_dir, config)
    evaluator.evaluate(data, test_data, config)


def newEnsemble(config):
    data = dataset.prepare_data(config.dataset_dir, config)
    test_data = dataset.prepare_test_data(config.test_dataset_dir, config)

    val_preds = []

    for model_type in config.model_types:
        train_accuracy = []
        val_accuracy = []
        train_preds = []

        for fold in range(5):
            res = evaluator.getPredictions(model_type, data, config, fold)
            train_accuracy.append(res[0])
            val_accuracy.append(res[1])
            train_preds.append(res[2])
            val_preds.append(res[3])

    res = evaluator.getIndividualEnsembleAccuracy(data, val_preds, config)

    # pdb.set_trace()

def mainContrastiveLoss(config):
    # Create save directories
    utils.create_directories(config)

    # Prepare and load the data
    if 'silences' in config.model_types:
        data = dataset.prepare_data_new(config.dataset_dir, config)
    else:
        data = dataset.prepare_data(config.dataset_dir, config)
    # print(data)
    # return
    # Train the ensemble models
    if config.training_type == 'bagging':
        ensemble_trainer.bagging_ensemble_training(data, config)
    elif config.training_type == 'boosting':
        ensemble_trainer.boosted_ensemble_training(data, config)

    # Evaluate the model
    if 'silences' not in config.model_types:
        test_data = dataset.prepare_test_data(config.test_dataset_dir, config)
        evaluator.evaluate(data, test_data, config)



if __name__ == '__main__':
    if sys.argv[1] == "train":
        main(config)
    elif sys.argv[1] == "evaluate":
        print("is Evaluating")
        mainEvaluate(config)
    elif sys.argv[1] == "newEnsemble":
        newEnsemble(config)
