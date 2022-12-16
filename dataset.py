import numpy as np
import os
from os.path import join
from os import listdir
from pandas import (
    DataFrame, 
    read_csv,
    )

class dataset_model:
    def __init__(self,
            params: dict) -> None:
        self.params = params
        self.data = None
        self._read()

    def _read(self) -> DataFrame:
        filename = self.params['path']
        data = read_csv(
            filename,
            header = None,
            encoding ='utf-8',
        )
        self.data = data

    def get_data(self) -> DataFrame:
        """
        Documentation
        """
        return self.data.copy()

        
