import dataclasses
from pathlib import Path

import pandas as pd

from feature.plot.plot_dataset import PlotTestImageDataset
from model.cnn import VGG16Wrapper
from utils.data.eegdataloader import EegDataSet


class PlotPredictor(object):

    def __init__(self, model_wrapper: VGG16Wrapper, plot_root: Path, save_root: Path, transformers=None):
        self.dataset = PlotTestImageDataset(plot_root, transformers)
        self.model_wrapper = model_wrapper
        self.save_root = save_root

    def __call__(self, batch_size, *args, **kwargs):
        predicted_array = self.model_wrapper.predict(self.dataset, batch_size)
        predicted_df = pd.DataFrame(predicted_array, columns=EegDataSet.EVENT_COLUMNS)
        predicted_meta = pd.DataFrame.from_records(
            [dataclasses.asdict(window_meta) for window_meta in self.dataset.window_metas])
        predicted_df = pd.concat([predicted_meta, predicted_df], axis=1)
        predicted_df.to_csv(self.save_root.joinpath("predicted.csv"), index=None)
