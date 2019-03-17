import dataclasses
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclasses.dataclass
class EegSeries:
    subject: int
    series: int
    data_df: pd.DataFrame

    def __post__init__(self):
        self.size = self.data_df.shape[0]


@dataclasses.dataclass
class TrainEegSeries(EegSeries):
    subject: int
    series: int
    data_df: pd.DataFrame
    event_df: pd.DataFrame

    #
    # def to_trials(self):
    #
    #     pass

    def crop(self, window_size, step_size, release_raw=False):
        n_windows = self.size // step_size if self.size % step_size else self.data_df // step_size - 1
        pad_size = (step_size * n_windows + window_size) - self.data_df.shape[0]

        cropped = [(self.data_df.iloc[i * step_size: i * step_size + window_size].values,
                    self.event_df.iloc[i * step_size: i * step_size + window_size].values
                    ) if i < n_windows - 1
                   else (self.data_df[i * step_size:],
                         self.event_df[i * step_size:])
                   for i in range(n_windows)]
        if pad_size:
            cropped[n_windows - 1] = (
                np.pad(cropped[n_windows - 1][0],
                       ((0, pad_size), (0, 0)), constant_values=0, mode="constant"),
                np.pad(cropped[n_windows - 1][1],
                       ((0, pad_size), (0, 0)), constant_values=0, mode="constant")
            )
        if release_raw:
            self.data_df = None
            self.event_df = None
        return cropped


@dataclasses.dataclass
class EegDataSet:
    subjects: List[List[TrainEegSeries]]
    N_EVENTS = 6

    def __post_init__(self):
        self.data_columns = list(self.subjects[0][0].data_df.columns)
        self.data_columns.remove("id")
        self.event_columns = list(self.subjects[0][0].event_df.columns)
        self.event_columns.remove("id")
        self.size = self.subjects[0][0].data_df.shape[0]

    @staticmethod
    def to_feature_matrix(self, release_df=True):
        matrix = []
        for subject in self.dataset.subjects:
            for eeg_series in subject:
                matrix.append(eeg_series.data_df.drop("id", axis=1).values)
                if release_df:
                    eeg_series.data_df = None
        return matrix

    @staticmethod
    def to_Y(self, release_df=True):
        matrix = []
        for subject in self.dataset.subjects:
            for eeg_series in subject:
                matrix.append(eeg_series.event_df.drop("id", axis=1).values)
                if release_df:
                    eeg_series.data_df = None
        return matrix


class EegDataLoader(object):
    FILE_PATH_PATTERN = re.compile(r"subj(\d+)_series(\d+)_\w+\.csv")
    FILE_SUBJECT_PATTERN_STR = r"subj{}_series(\d+)_\w+\.csv"

    def __init__(self, is_train=True, is_debug=False):
        self.data_root = Path("../input")
        if not self.data_root.exists():
            self.data_root = Path(__file__).parent.parent.parent.joinpath("data/raw")
        if is_debug:
            self.data_root = Path(__file__).parent.parent.parent.joinpath("data/debug")
        self.train_data_path = self.data_root.joinpath("train")
        self.test_data_path = self.data_root.joinpath("test")

        self.data_paths, self.event_paths = self.get_paths()
        self.subjects = self.parse_subjects()

        self._current_pairs = 0
        self._max_paris = len(self.data_paths)

    def __call__(self, *args, **kwargs):

        dataset = EegDataSet([[
            TrainEegSeries(series, subject, self.read_csv(data_path),
                           self.read_csv(event_path)) for
            series, (data_path, event_path) in enumerate(zip(data_paths, event_paths))]
            for subject, (data_paths, event_paths) in enumerate(zip(self.subj_datapaths, self.subj_evnet))
        ])
        return dataset

    @staticmethod
    def read_csv(path):
        if os.name == "nt":
            return pd.read_csv(path, encoding="utf-8", engine="python")
        return pd.read_csv(path, encoding="utf-8")

    def to_structured_paths(self):
        self.subj_datapaths = [
            sorted(filter(lambda x: x.name.startswith("subj" + str(subj) + "_"), self.data_paths),
                   key=lambda path: self.FILE_PATH_PATTERN.match(path.name).group(2))
            for subj in self.subjects
        ]

        self.subj_evnet = [
            sorted(filter(lambda x: x.name.startswith("subj" + str(subj) + "_"), self.event_paths),
                   key=lambda path: self.FILE_PATH_PATTERN.match(path.name).group(2))
            for subj in self.subjects
        ]

    def parse_subjects(self):
        return list(sorted({
            int(self.FILE_PATH_PATTERN.match(data_path.name).group(1)) for data_path in self.data_paths
        }))

    def get_paths(self):
        return self._to_csv_paths(self.train_data_path)

    def generator(self):
        while True:
            if self._current_pairs >= self._max_paris:
                break
            data_path = self.data_paths[self._current_pairs]
            event_path = self.event_paths[self._current_pairs]
            subject = self.FILE_PATH_PATTERN.match(data_path.name).group(1)
            series = self.FILE_PATH_PATTERN.match(event_path.name).group(2)
            yield TrainEegSeries(subject, series, pd.read_csv(data_path, encoding="utf-8"),
                                 pd.read_csv(event_path, encoding="utf-8"))
            self._current_pairs += 1

    @staticmethod
    def _to_csv_paths(root: Path):
        return (list(sorted(root.glob('*data.csv'))),
                list(sorted(root.glob('*events.csv'))))


class TestEegDataLoader(EegDataLoader):

    def __init__(self):
        super().__init__(False, False)

    def __call__(self, *args, **kwargs):

        if os.name == "nt":
            read_csv = lambda path: pd.read_csv(path, encoding="utf-8", engine="python")
        else:
            read_csv = lambda path: pd.read_csv(path, encoding="utf-8")

        dataset = EegDataSet([[
            EegSeries(series, subject, pd.read_csv(data_path, encoding="utf-8")) for
            series, data_path in enumerate(data_paths)]
            for subject, data_paths in enumerate(self.subj_datapaths)
        ])
        return dataset

    def to_structured_paths(self):
        self.subj_datapaths = [
            sorted(filter(lambda x: x.name.startswith("subj" + str(subj) + "_"), self.data_paths),
                   key=lambda path: self.FILE_PATH_PATTERN.match(path.name).group(2))
            for subj in self.subjects
        ]

    def generator(self):
        while True:
            if self._current_pairs >= self._max_paris:
                break
            data_path = self.data_paths[self._current_pairs]
            subject = self.FILE_PATH_PATTERN.match(data_path.name).group(1)
            series = self.FILE_PATH_PATTERN.match(data_path.name).group(2)
            yield EegSeries(subject, series, pd.read_csv(data_path, encoding="utf-8"))
            self._current_pairs += 1

    def get_paths(self):
        return self._to_csv_paths(self.test_data_path)
