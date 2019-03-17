import re
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from google.cloud import storage

from feature.plot.to_images import PlotWriter
from utils.data.eegdataloader import EegDataLoader

gs_client = storage.Client()
LABEL_FILE_NAME = "event_labels.csv"


class PlotLabeler(object):
    PLOT_PATH_PATTERN = re.compile(PlotWriter.FILE_PATH_PATTERN.format(
        "(?P<subject>\\d+)", "(?P<series>\\d+)", "(?P<window>\\d+)", "(?P<start>\\d+)", "(?P<end>\\d+)"))
    PLOT_ROOT_DIR_PATTERN = re.compile(r"window_(\d+)_stride_(\d+)")

    def __init__(self):
        self.label_df = None
        self._event_dfs = {}
        self._window_size = None
        self._stride_size = None

    def __call__(self, train_plots_root: Union[Path, str], train_raw_root: Path, n_jobs=None):
        print("labeling {}...".format(train_plots_root))
        plots_local_root = train_plots_root
        if str.startswith(str(train_plots_root), "gs://"):
            bucket = gs_client.get_bucket(train_plots_root.split("/")[0])
            # TODO

        plots_local_root = Path(plots_local_root)
        assert plots_local_root.exists()
        matched = self.PLOT_ROOT_DIR_PATTERN.match(plots_local_root.name)
        self._window_size = int(matched.group(1))
        self._stride_size = int(matched.group(2))
        event_paths = train_raw_root.glob("**/*events.csv")
        # print(list(event_paths))
        self._event_dfs = {}
        for path in event_paths:
            subject = EegDataLoader.FILE_PATH_PATTERN.match(path.name).group(1)
            series = EegDataLoader.FILE_PATH_PATTERN.match(path.name).group(2)
            if not subject in self._event_dfs:
                self._event_dfs[subject] = {}
            if not series in self._event_dfs[subject]:
                self._event_dfs[subject][series] = {}

            self._event_dfs[subject][series] = EegDataLoader.read_csv(path)
        self.event_columns = list(self._event_dfs["1"]["1"].columns)
        self.event_columns.remove("id")

        plot_paths = plots_local_root.glob("**/*.png")
        plot_files = [path.name for path in plot_paths]

        with Pool(n_jobs) as pool:
            rows = pool.map(self._label_window, plot_files)

        self.label_df = pd.DataFrame(columns=["plot_file"] + self.event_columns)
        self.label_df["plot_file"] = [row[0] for row in rows]
        self.label_df[self.event_columns] = np.vstack([row[1] for row in rows])

        print("labeling done")
        return self.label_df

    def _label_window(self, file_name):
        matched = self.PLOT_PATH_PATTERN.match(file_name)
        subject = matched.group("subject")
        series = matched.group("series")
        window = int(matched.group("window"))
        start = int(matched.group("start"))
        end = int(matched.group("end"))

        predict_start = window * self._stride_size + 1
        predict_end = (window + 1) * self._stride_size
        event_df = self._event_dfs[subject][series]
        if predict_start == event_df.shape[0]:
            return file_name, 0
        elif predict_end <= event_df.shape[0]:
            window_events = event_df[predict_start:predict_end][self.event_columns].values
        else:
            window_events = event_df[predict_start:][self.event_columns].values
        window_events = np.any(window_events, axis=0)

        return file_name, window_events


def write_labels_for_window_plot(train_plots_root: str, train_raw_root: str, n_jobs=None):
    df = PlotLabeler()(train_plots_root, Path(train_raw_root), n_jobs)
    df.to_csv(Path(train_plots_root).joinpath(LABEL_FILE_NAME), index=None)
