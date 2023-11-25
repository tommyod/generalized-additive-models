#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd

DATASET_DIRECTORY, _ = os.path.split(os.path.realpath(__file__))


def load_salaries():
    """Load salaries dataset.



    - https://www.kode24.no/artikkel/dykk-ned-i-kode24s-lonnstall-vis-oss-hva-du-lager/79548382
    - https://www.kaggle.com/datasets/olemagnushiback/lonn-data

    """
    return pd.read_csv(os.path.join(DATASET_DIRECTORY, "salaries.csv"))
