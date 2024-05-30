

def MLP_ids(dataset):
    hps = {
        'ACSIncome': {

        },
        'BankMarketing': {

        },
        'CreditDefault': {
            0: 6,
            0.01: 10,
            0.05: 37,
            0.1: 14,
            0.2: 37,
            0.4: 0,
            0.6: 41,
            0.8: 41,
            1.0: 41, # completely arbitrary
        },
        'HMDA': {

        },
        'MEPS': {
            0: 11,
            0.01: 129,
            0.05: 3,
            0.1: 18,
            0.2: 115,
            0.4: 131,
            0.6: 20,
        },
    }

    # replace string keys with float keys
    return_dict = {}
    for key in hps[dataset].keys():
        return_dict[key] = hps[dataset][key]
    if len(return_dict) == 0:
        raise ValueError(f'No hyperparameters found for dataset: {dataset}')

    return return_dict


def SimpleModel_ids(dataset, model):
    hps = {
        'LogisticRegression': {
            'ACSIncome': {
                0: 0,
                0.01: 1,
                0.05: 0,
                0.1: 2,
                0.2: 2,
                0.4: 0,
                0.6: 2,
            },
            'BankMarketing': {
                0: 2,
                0.01: 1,
                0.05: 2,
                0.1: 0,
                0.2: 0,
                0.4: 0,
                0.6: 3,
            },
            'CreditDefault': {
                0: 3,
                0.01: 3,
                0.05: 3,
                0.1: 3,
                0.2: 2,
                0.4: 3,
                0.6: 3,
            },
            'HMDA': {
                0: 3,
                0.01: 3,
                0.05: 0,
                0.1: 0,
                0.2: 0,
                0.4: 0,
                0.6: 3,
            },
            'MEPS': {
                0: 0,
                0.01: 2,
                0.05: 1,
                0.1: 0,
                0.2: 2,
                0.4: 0,
                0.6: 2,
            },
        },
        'NaiveBayes': {
            # no hyperparamters
            dataset: {
                0: -1,
                0.01: -1,
                0.05: -1,
                0.1: -1,
                0.2: -1,
                0.4: -1,
                0.6: -1,
            } for dataset in ['ACSIncome', 'CreditDefault', 'BankMarketing', 'MEPS', 'HMDA']
        },
        'SVM': {
            'ACSIncome': {
                0: 0,
                0.01: 1,
                0.05: 3,
                0.1: 1,
                0.2: 1,
                0.4: 1,
                0.6: 3,
            },
            'BankMarketing': {
                0: 3,
                0.01: 3,
                0.05: 0,
                0.1: 1,
                0.2: 0,
                0.4: 1,
                0.6: 3,
            },
            'CreditDefault': {
                0: 1,
                0.01: 2,
                0.05: 1,
                0.1: 1,
                0.2: 3,
                0.4: 0,
                0.6: 1,
            },
            'HMDA': {
                0: 0,
                0.01: 0,
                0.05: 1,
                0.1: 3,
                0.2: 0,
                0.4: 1,
                0.6: 0,
            },
            'MEPS': {
                0: 3,
                0.01: 3,
                0.05: 3,
                0.1: 3,
                0.2: 3,
                0.4: 3,
                0.6: 3,
            },
        },
        'DecisionTree': {
            'ACSIncome': {
                0: 5,
                0.01: 5,
                0.05: 3,
                0.1: 5,
                0.2: 5,
                0.4: 4,
                0.6: 5,
            },
            'BankMarketing': {
                0: 3,
                0.01: 4,
                0.05: 5,
                0.1: 5,
                0.2: 5,
                0.4: 3,
                0.6: 5,
            },
            'CreditDefault': {
                0: 3,
                0.01: 4,
                0.05: 5,
                0.1: 4,
                0.2: 5,
                0.4: 3,
                0.6: 4,
            },
            'HMDA': {
                0: 5,
                0.01: 3,
                0.05: 5,
                0.1: 4,
                0.2: 3,
                0.4: 4,
                0.6: 3,
            },
            'MEPS': {
                0: 5,
                0.01: 5,
                0.05: 4,
                0.1: 5,
                0.2: 5,
                0.4: 5,
                0.6: 3,
            },
        },
        'RandomForest': {
            'ACSIncome': {
                0: 8,
                0.01: 7,
                0.05: 8,
                0.1: 8,
                0.2: 8,
                0.4: 8,
                0.6: 8,
            },
            'BankMarketing': {
                0: 2,
                0.01: 1,
                0.05: 2,
                0.1: 8,
                0.2: 2,
                0.4: 8,
                0.6: 1,
            },
            'CreditDefault': {
                0: 4,
                0.01: 4,
                0.05: 5,
                0.1: 4,
                0.2: 3,
                0.4: 3,
                0.6: 5,
            },
            'HMDA': {
                0: 7,
                0.01: 8,
                0.05: 11,
                0.1: 8,
                0.2: 2,
                0.4: 8,
                0.6: 2,
            },
            'MEPS': {
                0: 2,
                0.01: 7,
                0.05: 1,
                0.1: 2,
                0.2: 3,
                0.4: 1,
                0.6: 8,
            },
        },
    }

    # replace string keys with float keys
    return_dict = {}
    for key in hps[model][dataset].keys():
        return_dict[key] = hps[model][dataset][key]
    if len(return_dict) == 0:
        raise ValueError(f'No hyperparameters found for dataset: {dataset}')

    return return_dict


def ResNet_ids(dataset):
    hps = {
        'YelpPolarity': {
            0: {
                'ERM': [({}, 1)],
            },
        },
        'AmazonPolarity': {
            0: {
                'ERM': [({}, 3)],
            },
        },
        'CivilComments': {
            0: {
                'ERM': [({}, 0)],
            },
        },
    }

    # replace string keys with float keys
    return_dict = {}
    for key in hps[dataset].keys():
        return_dict[key] = hps[dataset][key]
    if len(return_dict) == 0:
        raise ValueError(f'No hyperparameters found for dataset: {dataset}')

    return return_dict
