from json import load
from os import makedirs

def get_params() -> dict:
    params = {
    "path": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
     #"dataset": "/iris.data" 
    }
    return params


def mkdir(path: str) -> None:
    """
    Documentation
    """
    makedirs(
        path,
        exist_ok=True
        )
