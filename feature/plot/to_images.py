import argparse

import matplotlib as mpl

mpl.use('Agg')
import multiprocessing as mp
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.data.eegdataloader import EegDataSet, EegDataLoader, TestEegDataLoader, EegSeries


class PlotWriter(object):
    FILE_PATH_PATTERN = "subject_{}_series_{}_window_{}_start_{}_end_{}.png"
    WINDOW_ROOT_DIR = "window_{}_stride_{}"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self._save_dir = None
        self._dataset: EegDataSet = None
        self._figsize = None
        self._current_save_dir = None
        self.n_jobs = -1

    # def write(self, dataset: DataSet, window_size, stride_size, figsize, n_jobs=None):
    def write(self, datagenerator, window_size, stride_size, figsize, n_jobs=None):
        # self._dataset = dataset
        self._datagenerator = datagenerator
        self._save_dir = self.root_dir.joinpath(self.WINDOW_ROOT_DIR.format(window_size, stride_size))
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._window_size = window_size
        self._stride_size = stride_size
        self._figsize = figsize
        self.n_jobs = n_jobs
        self._current_df = None
        self._subject = None
        self._series = None
        # self.data_columns = self._dataset.data_columns

        for eeg_series in datagenerator():
            print("writing plot of windows from subject {} series {}".format(eeg_series.subject, eeg_series.series))
            self.data_columns = list(eeg_series.data_df.columns)
            self.data_columns.remove("id")
            self._write_series(eeg_series)

    def _write_series(self, eeg_series: EegSeries):
        self._current_save_dir = self._save_dir.joinpath(str(eeg_series.subject)).joinpath(str(eeg_series.series))
        self._current_save_dir.mkdir(parents=True, exist_ok=True)
        self._subject = eeg_series.subject
        self._series = eeg_series.series

        # self._current_df = self._pad_df_with_first_value(eeg_series.data_df, self._window_size - 1)
        self._current_df = self._pad_df_with_first_value(eeg_series.data_df, self._window_size)
        indices = [(i, self._stride_size * i, self._stride_size * i + self._window_size)
                   for i in range((self._current_df.shape[0] - self._window_size + 1) // self._stride_size)]
        with mp.get_context("spawn").Pool(self.n_jobs) as pool:
            # tqdm(pool.imap_unordered(self._write_window, indices))
            pool.map(self._write_window, indices)

    @staticmethod
    def _pad_df_with_first_value(df, pad_size):
        return pd.concat([df.iloc[0:1] for _ in range(pad_size)] + [df], axis=0, ignore_index=True)

    def _write_window(self, window_start_end_idx):
        window_idx, start, end = window_start_end_idx

        fig = plt.figure(random.randint(0, 100))
        dpi = fig.get_dpi()
        fig.set_size_inches(self._figsize[0] / dpi, self._figsize[1] / dpi)

        axes = self._current_df.iloc[start:end][self.data_columns].plot(
            subplots=True, sharex=True, sharey=True,
            figsize=(self._figsize[0] / dpi, self._figsize[1] / dpi)
        )
        for ax in axes:
            ax.axis("off")
            ax.legend().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(self._current_save_dir.joinpath(self.FILE_PATH_PATTERN.format(self._subject, self._series,
                                                                                  window_idx, start, end)),
                    bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0.0)
        plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", default=False, type=bool)
    parser.add_argument("--window", default=750, type=int)
    parser.add_argument("--stride", default=75, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n_jobs", default=None, type=int)

    args = parser.parse_args()

    is_train = not args.test
    root = Path("output/features").joinpath("train" if is_train else "test")
    writer = PlotWriter(root)

    if is_train:
        dataloader = EegDataLoader(is_train=is_train, is_debug=args.debug)
    else:
        dataloader = TestEegDataLoader()

    writer.write(dataloader.generator, args.window, args.stride, figsize=(510, 510), n_jobs=args.n_jobs)
