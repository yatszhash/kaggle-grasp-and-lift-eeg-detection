import dataclasses
import random
from pathlib import Path

import skimage.io
import torch.utils.data
from pandas import DataFrame

from feature.plot import label_window
from feature.plot.label_window import LABEL_FILE_NAME
from feature.preprocess import VGG16Transformers
from utils.data.eegdataloader import EegDataLoader

RANDOM_STATE = 10


@dataclasses.dataclass
class Window:
    subject: int
    series: int
    start: int
    end: int
    target_frame_start: int
    target_frame_end: int
    path: Path


class PlotTestImageDataset(torch.utils.data.Dataset):

    def __init__(self, plot_root: Path, transformers=VGG16Transformers()):
        self.load_plot_paths(plot_root)
        self.transformers = transformers
        self.plot_paths = self.load_plot_paths(plot_root)
        self.window_size, self.stride_size = self.to_window_stride(self.plot_paths[0])
        self.window_metas = [self.to_window(path) for path in self.plot_paths]

    @staticmethod
    def load_plot_paths(plot_root):
        assert plot_root.exists()
        plot_paths = plot_root.glob("**/*.png")
        plot_paths = sorted(plot_paths,
                            key=lambda x: int(
                                label_window.PlotLabeler.PLOT_PATH_PATTERN.match(x.name).group(2)))
        plot_paths = list(sorted(plot_paths,
                                 key=lambda x: int(
                                     label_window.PlotLabeler.PLOT_PATH_PATTERN.match(x.name).group(1))))
        return plot_paths

    def to_window(self, plot_path):
        matched = label_window.PlotLabeler.PLOT_PATH_PATTERN.match(plot_path.name)
        start = int(matched.group("start"))
        end = int(matched.group("end"))
        target_frame_start = end
        target_frame_end = end + self.window_size

        return Window(int(matched.group("subject")), int(matched.group("series")),
                      start, end, target_frame_start, target_frame_end, plot_path)

    @staticmethod
    def to_window_stride(plot_path):
        window_config = plot_path.parent.parent.parent.name
        matched = label_window.PlotLabeler.PLOT_ROOT_DIR_PATTERN.match(window_config)
        return int(matched.group(1)), int(matched.group(2))

    def __getitem__(self, index):
        sample = skimage.io.imread(str(self.plot_paths[index]))[:, :, :3]
        if not self.transformers:
            return sample
        return self.transformers(sample)

    def __len__(self):
        return len(self.plot_paths)


class PlotTrainValidDataSets(object):

    def __init__(self, plot_root: Path, valid_n_subjects=1, transformers=None):
        self.plot_paths = PlotTestImageDataset.load_plot_paths(plot_root)
        self.valid_n_subjects = valid_n_subjects
        self.label_df = EegDataLoader.read_csv(plot_root.joinpath(LABEL_FILE_NAME))
        self.label_df["subject"] = self.label_df["plot_file"].apply(lambda x: int(
            label_window.PlotLabeler.PLOT_PATH_PATTERN.match(x).group(1)))
        self.label_df["series"] = self.label_df["plot_file"].apply(lambda x: int(
            label_window.PlotLabeler.PLOT_PATH_PATTERN.match(x).group(2)))
        self.label_df.sort_values(by=["subject", "series"], inplace=True)
        self.label_df.reset_index(inplace=True)
        assert self.label_df["plot_file"][0] == self.plot_paths[0].name
        assert self.label_df["plot_file"][self.label_df.shape[0] - 1] == self.plot_paths[-1].name
        assert self.label_df["plot_file"][66] == self.plot_paths[66].name

        self.subjects = self.label_df.subject.unique().tolist()

        shuffled_subjects = [s for s in self.subjects]
        random.seed(RANDOM_STATE)
        random.shuffle(shuffled_subjects)
        self.train_subjects = shuffled_subjects[valid_n_subjects:]
        self.valid_subjects = shuffled_subjects[:valid_n_subjects]

        print("train subject: {} \nvalid subjects: {}".format(str(self.train_subjects),
                                                              str(self.valid_subjects)))

        train_label_df = self.label_df[self.label_df["subject"].isin(self.train_subjects)]
        valid_label_df = self.label_df[self.label_df["subject"].isin(self.valid_subjects)]

        train_plot_paths = [self.plot_paths[idx] for idx in train_label_df.index.tolist()]
        valid_plot_paths = [self.plot_paths[idx] for idx in valid_label_df.index.tolist()]
        self.train_dataset = PlotTrainImageDataset(train_plot_paths, train_label_df, transformers)
        self.valid_dataset = PlotTrainImageDataset(valid_plot_paths, valid_label_df, transformers)


class PlotTrainImageDataset(torch.utils.data.Dataset):

    def __init__(self, plot_paths, label_df: DataFrame, transformers):
        self.label_df = label_df
        self.plot_paths = plot_paths
        self.transformers = transformers

    def __getitem__(self, index):
        image = skimage.io.imread(str(self.plot_paths[index]))[:, :, :3]

        if not self.transformers:
            return image
        return {"image": self.transformers(image),
                "label": self.label_df.iloc[index, 2:-2].values.astype("float32")}

    def __len__(self):
        return self.label_df.shape[0]


