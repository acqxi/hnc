import glob
import os
from numbers import Number
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


class ExcelDataLoader:
    def __init__(
        self,
        path,
        indexName: Optional[str] = "病人編號",
        index2Name: Optional[str] = "CT日期",
        lapName: Optional[str] = "LND-L",
        header: int = 1,
        dropNaRowName: Optional[str] = None,
        fillNaWith: Optional[Union[int, float, str]] = 0,
    ) -> None:
        self.df = pd.read_excel(path, header=header)
        self.index_name = indexName
        self.index2_name = index2Name
        self.lap_name = lapName
        self.cols = self.df.columns
        if dropNaRowName:
            self.df = self.df.drop(np.where(self.df[dropNaRowName].isna())[0])
        if fillNaWith is not False and fillNaWith is not None:
            self.df = self.df.fillna(fillNaWith)

    def __getitem__(self, index):
        if (
            isinstance(index, Iterable)
            and sum([isinstance(i, (Number, str)) for i in index]) == 4
        ):
            i, ii, j, k = [i for i in index if isinstance(i, (Number, str))]
            try:
                return self.df[
                    ((self.df[self.index_name] == i) & (self.df[self.index2_name] == ii)) & (self.df[self.lap_name] == j)
                ][k].item()
            except ValueError as e:
                return self.df[
                    ((self.df[self.index_name] == i) & (self.df[self.index2_name] == ii)) & (self.df[self.lap_name] == j)
                ][k]
        else:
            return self.df[index]