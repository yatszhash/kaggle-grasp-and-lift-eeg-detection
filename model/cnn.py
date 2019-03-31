import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.data.eegdataloader import EegDataSet

RANDOM_SEED = 10

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


@dataclasses.dataclass
class Params(object):
    separate_temporal_kernel_size = 4
    dilation = (1, 2)
    separate_temporal_channel = 8
    separate_spatial_channel = 16
    all_spatial_kernel_size = 32
    first_pool_size = 3


class VGG16Wrapper(object):

    def __init__(self, dropout_rate, save_dir: Path, optimizer_factory=None, loss_function=None, score_function=None,
                 lr=2e-3, model_pickled=None):
        vgg16 = torchvision.models.vgg16(pretrained=True)

        for param in vgg16.parameters():
            param.required_grad = False

        in_channel = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(in_channel, 32)

        self.model = nn.Sequential(
            vgg16,
            nn.Dropout(dropout_rate),
            nn.Linear(32, EegDataSet.N_EVENTS)
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            self.model = torch.nn.DataParallel(self.model).cuda()

        if model_pickled:
            self.model.load_state_dict(torch.load(str(model_pickled)))

        self.n_epoch = None
        self._current_epoch = 0
        self._current_max_valid_score = 0
        self._early_stop_count = 0

        self.save_path = save_dir.joinpath("vgg16_pretrained")
        self.train_result_path = save_dir.joinpath("result.csv")
        self.train_results = pd.DataFrame()

        if not optimizer_factory:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = optimizer_factory(self.model)

        if loss_function:
            self.loss_function = loss_function
        else:
            self.loss_function = nn.BCEWithLogitsLoss()

        if score_function:
            self.score_function = score_function
        else:
            self.score_function = roc_auc_score

    def train(self, train_dataset: Dataset, valid_dataset: Dataset, n_epochs, train_batch_size,
              valid_batch_size, patience=10, num_workers=0):
        self.clear_history()
        self.patience = patience
        self._train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                            num_workers=num_workers)
        self._valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True,
                                            num_workers=num_workers)
        self.n_epoch = n_epochs

        logger.info("train with data size: {}".format(len(self._train_dataloader.dataset)))
        logger.info("valid with data size: {}".format(len(self._valid_dataloader.dataset)))

        iterator = tqdm(range(n_epochs))
        for epoch in iterator:
            self._current_epoch = epoch + 1
            logger.info("training %d  / %d epochs", self._current_epoch, n_epochs)

            self._train_epoch()
            self.write_current_result()
            self._valid_epoch()
            self.write_current_result()

            valid_score = self.train_results["valid_score"][self._current_epoch]
            if valid_score <= self._current_max_valid_score:
                self._early_stop_count += 1
            else:
                logger.info("validation score is improved from %.3f to %.3f",
                            self._current_max_valid_score, valid_score)
                self._current_max_valid_score = valid_score
                self._early_stop_count = 0
                self.save_models()

            if self._early_stop_count >= self.patience:
                logger.info("======early stopped=====")
                self.model.load_state_dict(torch.load(self.save_path))
                iterator.close()
                break

        logger.info("train done!")
        return self._current_max_valid_score

    def write_current_result(self):
        self.train_results.to_csv(self.train_result_path, encoding="utf-8")

    def clear_history(self):
        self.n_epoch = None
        self._current_epoch = 0

        self.train_losses = []
        self.train_scores = []
        self.valid_losses = []
        self.valid_scores = []

        self._current_max_valid_score = 0
        self._early_stop_count = 0

    def _train_epoch(self):
        self.model.train()

        all_labels = []
        all_outputs = []
        total_loss = 0.0
        for i, data in enumerate(self._train_dataloader):
            inputs = data["image"]
            # print("batch data size {}".format(inputs.size()))

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            labels = data["label"]
            all_labels.append(labels.cpu().detach().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

            labels = labels.to(device)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.cpu().detach().item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self._train_dataloader)
        logger.info("******train loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_results.loc[self._current_epoch, "train_loss"] = avg_loss
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        score = self.score_function(all_labels, all_outputs)
        logger.info("******train score at epoch %d: %.3f :" % (self._current_epoch, score))
        self.train_results.loc[self._current_epoch, "train_score"] = score

    def _valid_epoch(self):
        total_loss = 0.0

        all_labels = []
        all_outputs = []
        self.model.eval()
        for i, data in enumerate(self._valid_dataloader):
            inputs = data["image"]

            outputs = self.model(inputs)
            labels = data["label"]
            all_labels.append(labels.cpu().detach().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())
            labels = labels.to(device)
            loss = self.loss_function(outputs, labels)

            total_loss += loss.cpu().detach().item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] validation loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self._valid_dataloader)
        logger.info("******valid loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_results.loc[self._current_epoch, "valid_loss"] = avg_loss

        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        score = self.score_function(all_labels, all_outputs)
        logger.info("******valid score at epoch %d: %.3f :" % (self._current_epoch, score))
        self.train_results.loc[self._current_epoch, "valid_score"] = score

    def save_models(self):
        torch.save(self.model.state_dict(), str(self.save_path))
        logger.info("Checkpoint saved")

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()

    def predict(self, dataset: Dataset, batch_size):
        logger.info("predicting %d samples...".format(len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        return np.vstack([torch.sigmoid(self.model(x)).cpu().detach().numpy() for x in tqdm(dataloader)])
