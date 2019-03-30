# #====================== for debug===============
# sys.path.append(str(Path(__file__).parent.parent.parent.parent.joinpath("pycharm-debug-py3k.egg")))
# import pydevd
#
# pydevd.settrace('local-dev', port=12345, stdoutToServer=True,
# stderrToServer=True)
# #===============================================
import logging
import os
import sys
from pathlib import Path

from feature.plot.prediction import PlotPredictor
from feature.preprocess import VGG16Transformers
from model.cnn import VGG16Wrapper

n_cpus = os.cpu_count()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


def main():
    debug = False
    # datasets = PlotTrainValidDataSets(plot_root_path, valid_n_subjects=2, transformers=VGG16Transformers())
    save_dir = Path("/mnt/gcs/kaggle-grasp-and-lift-eeg-detection/model/vgg_pretrained/window_750_stride_75_lr2e-4")
    save_dir.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(str(save_dir.joinpath("predict.log")))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    model_wrapper = VGG16Wrapper(dropout_rate=0.5, save_dir=save_dir, lr=2e-4,
                                 model_pickled=save_dir.joinpath("vgg16_pretrained"))

    # if debug:
    #     DEBUG_SIZE = 4000
    #     datasets.train_dataset.label_df = datasets.train_dataset.label_df[:DEBUG_SIZE]
    #     datasets.train_dataset.plot_paths = datasets.train_dataset.plot_paths[:DEBUG_SIZE]
    #     datasets.valid_dataset.label_df = datasets.valid_dataset.label_df[:DEBUG_SIZE]
    #     datasets.valid_dataset.plot_paths = datasets.valid_dataset.plot_paths[:DEBUG_SIZE]
    # model_wrapper.train(datasets.train_dataset, datasets.valid_dataset,
    #                     train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10,
    #                     num_workers=n_cpus)
    #

    train_save_dir = save_dir.joinpath("prediction/train")
    train_save_dir.mkdir(exist_ok=True, parents=True)
    plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")
    predictor = PlotPredictor(model_wrapper, plot_root=plot_root_path, save_root=train_save_dir,
                              transformers=VGG16Transformers())
    predictor(batch_size=128)

    test_save_dir = save_dir.joinpath("prediction/test")
    test_save_dir.mkdir(exist_ok=True, parents=True)
    plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/test/window_750_stride_75")
    predictor = PlotPredictor(model_wrapper, plot_root=plot_root_path, save_root=test_save_dir,
                              transformers=VGG16Transformers())
    predictor(batch_size=128)


if __name__ == '__main__':
    main()
