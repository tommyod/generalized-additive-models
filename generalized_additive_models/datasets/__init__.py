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


def load_mcycle():
    """Load mcycle dataset.

    - https://search.r-project.org/CRAN/refmans/VarReg/html/mcycle.html
    - https://r-data.pmagunia.com/dataset/r-dataset-package-mass-mcycle

    """
    return pd.read_csv(os.path.join(DATASET_DIRECTORY, "mcycle.csv"))


def load_bicycles():
    """Load bicyles dataset.

    A subset of data from bike measurement in Stavanger, Norway.

    - https://opencom.no/dataset/samling-av-sykkelmalinger-stavanger/resource/ac59ce73-e691-430b-8619-83dbf637d861

    """
    df = pd.read_csv(os.path.join(DATASET_DIRECTORY, "bicycles.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_powerlifters():
    """Load powerlifters dataset.

    RAW (unequipped) results from Norwegian powerlifters in Norwegian meets.
    A subset of data from OpenPowerlifting.

    - https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html

    """
    df = pd.read_csv(os.path.join(DATASET_DIRECTORY, "powerlifters.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df
