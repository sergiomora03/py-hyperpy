#!/usr/bin/python3
#-*- coding: utf-8 -*-

# Author: Sergio A. Mora Pardo <sergiomora823@gmail.com>
# Project: hyperapy

import os
from io import StringIO
from zipfile import Path
import pandas as pd

class ExampleConfig:
    def __init__(self,workingDirectory:str=os.path.join(os.getcwd()+'\\docs\\data\\titanic.zip')) -> None:
        self.workingDirectory=workingDirectory

    def readData(self) -> pd.DataFrame:
        print(f'reading data from: {self.workingDirectory}')
        train=pd.read_csv(StringIO(Path(self.workingDirectory, at="train.csv").read_text()))
        train.dropna(inplace=True)
        test=pd.read_csv(StringIO(Path(self.workingDirectory, at="test.csv").read_text()))
        sub=pd.read_csv(StringIO(Path(self.workingDirectory, at="gender_submission.csv").read_text()))
        return train, test, sub

    seed=1
