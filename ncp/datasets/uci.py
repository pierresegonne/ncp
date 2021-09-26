import pathlib
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
UCI Datasets, with Dy = 1:
Boston
Concrete
Kin8nm
Power plant (CCPP)
Protein
Superconductivity
Wine-red
Wine-white
Yacht
"""


class UCIDataset(Enum):
    BOSTON = "boston"
    CCPP = "ccpp"
    CONCRETE = "concrete"
    KIN8NM = "kin8nm"
    PROTEIN = "protein"
    SUPERCONDUCT = "superconduct"
    WINE_RED = "wine_red"
    WINE_WHITE = "wine_white"
    YACHT = "yacht"


UCI_DATASETS_PATH = pathlib.Path(__file__).parent.resolve() / "uci"


def generate_and_save_one_uci_dataset(
    dataset: UCIDataset, inputs: np.ndarray, outputs: np.ndarray
) -> None:
    print(f"Generating {dataset.value}")
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, outputs, train_size=0.9
    )
    dataset_path = UCI_DATASETS_PATH / dataset.value
    with open(dataset_path / "-train-inputs.npy", "wb") as f:
        np.save(f, train_inputs)
    with open(dataset_path / "-train-targets.npy", "wb") as f:
        np.save(f, train_targets)
    with open(dataset_path / "-test-inputs.npy", "wb") as f:
        np.save(f, test_inputs)
    with open(dataset_path / "-test-targets.npy", "wb") as f:
        np.save(f, test_targets)
    print(" - OK")


def _generate_uci_from_csv(
    dataset: UCIDataset,
    filename: str,
    input_columns: Optional[List[str]],
    target_column: str,
) -> None:
    df = pd.read_csv(UCI_DATASETS_PATH / dataset.value / filename)
    # Split features, targets
    if input_columns is None:
        x = df.drop(columns=[target_column]).to_numpy()
    else:
        x = df[input_columns].to_numpy()
    y = df[target_column].to_numpy()
    generate_and_save_one_uci_dataset(dataset, x, y)


def generate_boston() -> None:
    DATA_FILENAME = "housing.data"
    """
    Link to get the housing dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    """
    dims = 13
    out_dims = 1
    df = pd.read_csv(UCI_DATASETS_PATH / UCIDataset.BOSTON.value / DATA_FILENAME)
    # Loads rows as string
    data = np.empty((len(df.index), dims + out_dims))
    for i in range(data.shape[0]):
        data[i] = np.array([float(el) for el in df.values[i][0].split(" ") if el != ""])
    x = data[:, :-1]
    y = data[:, -1]
    generate_and_save_one_uci_dataset(UCIDataset.BOSTON, x, y)


def generate_ccpp() -> None:
    DATA_FILENAME = "ccpp.csv"
    """
    Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
    Note that I converted the xls file to csv, removed the sheets and renamed and the file itself.
    """
    COLUMNS = ["AT", "V", "AP", "RH", "PE"]
    Y_LABEL = "PE"
    _generate_uci_from_csv(UCIDataset.CCPP, DATA_FILENAME, COLUMNS, Y_LABEL)


def generate_concrete() -> None:
    DATA_FILENAME = "concrete.csv"
    """
    Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    Note that I converted the xls file to csv, renamed the label column and the file itself.
    """
    COLUMNS = [
        "Cement (component 1)(kg in a m^3 mixture)",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
        "Fly Ash (component 3)(kg in a m^3 mixture)",
        "Water  (component 4)(kg in a m^3 mixture)",
        "Superplasticizer (component 5)(kg in a m^3 mixture)",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)",
        "Age (day)",
        "Concrete compressive strength(MPa)",
    ]
    Y_LABEL = "Concrete compressive strength(MPa)"
    _generate_uci_from_csv(UCIDataset.CONCRETE, DATA_FILENAME, COLUMNS, Y_LABEL)


def generate_kin8nm() -> None:
    DATA_FILENAME = "kin8nm.csv"
    """
    Link to get the kin8nm.csv file: https://www.openml.org/d/189
    """
    COLUMNS = [
        "theta1",
        "theta2",
        "theta3",
        "theta4",
        "theta5",
        "theta6",
        "theta7",
        "theta8",
    ]
    Y_LABEL = "y"
    _generate_uci_from_csv(UCIDataset.KIN8NM, DATA_FILENAME, COLUMNS, Y_LABEL)


def generate_protein() -> None:
    DATA_FILENAME = "CASP.csv"
    """
    Link to get the CASP.csv file: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure#
    """
    COLUMNS = [f"F{i}" for i in range(1, 10)]
    Y_LABEL = ["RMSD"]
    _generate_uci_from_csv(UCIDataset.PROTEIN, DATA_FILENAME, COLUMNS, Y_LABEL)


def generate_superconduct() -> None:
    DATA_FILENAME = "raw.csv"
    """
    Link to get the raw.csv data: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
    """
    Y_LABEL = "critical_temp"
    _generate_uci_from_csv(UCIDataset.SUPERCONDUCT, DATA_FILENAME, None, Y_LABEL)


def generate_wine_red() -> None:
    DATA_FILENAME = "winequality-red.csv"
    """
    Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Note that I changed the delimiter from ; to ,
    """
    COLUMNS = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
    Y_LABEL = "quality"
    _generate_uci_from_csv(UCIDataset.WINE_RED, DATA_FILENAME, COLUMNS, Y_LABEL)


def generate_wine_white() -> None:
    DATA_FILENAME = "winequality-white.csv"
    """
    Link to get the concrete.csv file: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Note that I changed the delimiter from ; to ,
    """
    COLUMNS = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
    Y_LABEL = "quality"
    _generate_uci_from_csv(UCIDataset.WINE_WHITE, DATA_FILENAME, COLUMNS, Y_LABEL)


def generate_yacht() -> None:
    DATA_FILENAME = "yacht_hydrodynamics.data"
    """
    Link to get the yacht_hydrodynamics.data file: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
    """
    COLUMNS = [
        "Longitudinal position of the center of the buoyancy",
        "Prismatic coefficient",
        "Length-displacement ratio",
        "Beam-draught ratio",
        "Length-beam ratio",
        "Froude number",
        "Residuary resistance per unit weight of displacement",
    ]
    Y_LABEL = "Residuary resistance per unit weight of displacement"
    df = pd.read_fwf(
        UCI_DATASETS_PATH / UCIDataset.YACHT.value / DATA_FILENAME, names=COLUMNS
    )
    # Split features, targets
    x = df.drop(columns=[Y_LABEL]).values
    y = df[Y_LABEL].values
    generate_and_save_one_uci_dataset(UCIDataset.YACHT, x, y)


def generate_uci_datasets() -> None:
    """
    Processes the uci datasets to generate the respective numpy datasets folders
    Format: folder with :/
    -train-inputs.npy
    -train-targets.npy
    -test-inputs.npy
    -test-targets.npy
    """
    for uci_dataset in UCIDataset:
        eval(f"generate_{uci_dataset.value}()")
    print("\n💯All UCI datasets are ready to be used with `load_numpy_dataset`\n")


if __name__ == "__main__":
    generate_uci_datasets()
