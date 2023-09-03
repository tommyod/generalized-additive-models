#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

file_path = os.path.realpath(__file__)


def load_salaries():
    """Load salaries dataset.



    - https://www.kode24.no/artikkel/dykk-ned-i-kode24s-lonnstall-vis-oss-hva-du-lager/79548382
    - https://www.kaggle.com/datasets/olemagnushiback/lonn-data

    """
    return pd.load_csv(os.path.join(file_path, "salaries.csv"))
