REQUIRED_FILES = {
    'Camelyon17': [
        'patches',
        'metadata.csv',
    ]
}



def req_files(dataset):
    return REQUIRED_FILES[dataset]