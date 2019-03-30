from torchvision import transforms

from utils.data.eegdataloader import EegDataSet


class NeighborAverageCorrector(object):
    neighbors = {
        "Fp1": ["Fp2", "F7", "F3", "Fz"],
        "Fp2": ["F8", "F4", "Fz", "Fp1"],
        "F7": ["Fp1", "F3", "FC5", "T7"],
        "F3": ["F7", "Fp1", "Fz", "FC1"],
        "Fz": ["F3", "FC1", "FC2", "F4"],
        "F4": ["Fz", "Fp2", "F8", "FC6", "FC2"],
        "F8": ["Fp2", "F4", "FC6", "T8"],
        "FC5": ["F7", "F3", "FC1", "C3", "T7"],
        "FC1": ["FC5", "F3", "Fz", "FC2", "Cz", "C3"],
        "FC2": ["FC1", "Fz", "F4", "FC6", "C4", "Cz"],
        "FC6": ["FC2", "F4", "F8", "T8", "C4"],
        "T7": ["F7", "FC5", "C3", "CP5", "P7", "TP9"],
        "C3": ["T7", "FC5", "FC1", "Cz", "CP1", "CP5"],
        "Cz": ["C3", "FC1", "FC2", "C4", "CP2", "CP1"],
        "C4": ["Cz", "FC2", "FC6", "T8", "CP6", "CP2"],
        "T8": ["C4", "FC6", "F8", "TP10", "CP6"],
        "TP9": ["T7", "CP5", "P7", "PO9"],
        "CP5": ["TP9", "T7", "C3", "CP1", "P3", "P7"],
        "CP1": ["CP5", "C3", "Cz", "CP2", "Pz", "P3"],
        "CP2": ["CP1", "Cz", "C4", "CP6", "P4", "Pz"],
        "CP6": ["CP2", "C4", "T8", "TP10", "P8", "P4"],
        "TP10": ["CP6", "T8", "P8"],
        "P7": ["TP9", "CP5", "O1", "PO9"],
        "P3": ["P7", "CP5", "CP1", "Pz", "O1"],
        "Pz": ["P3", "CP1", "CP2", "P4", "O2", "Oz", "O1"],
        "P4": ["Pz", "CP2", "CP6", "P8", "O2"],
        "P8": ["P4", "CP6", "TP10", "PO10", "O2"],
        "PO9": ["P7", "P3", "O1"],
        "O1": ["PO9", "P7", "P3", "Pz", "Oz"],
        "Oz": ["O1", "Pz", "O2"],
        "O2": ["Oz", "P4", "P8", "PO10", "Oz"],
        "PO10": ["O2", "P8"]
    }

    def transform(self, data_df):
        for col, nb in self.neighbors.items():
            data_df[col + "_nb_corrected"] = (data_df[col] - data_df[nb].mean(axis=1)).astype("float32")

    def __call__(self, datasets: EegDataSet):
        for subject in datasets.subjects:
            for eeg_series in subject:
                self.transform(eeg_series.data_df)
        return datasets


class DropColumns(object):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, df):
        df.drop(self.columns, axis=1, inplace=True)

    def __call__(self, datasets: EegDataSet):
        for subject in datasets.subjects:
            for eeg_series in subject:
                self.transform(eeg_series.data_df)
        return datasets


class MedianScaler(object):
    '''
    the baseline to 0
    '''

    def transform(self, data_df):
        point_cols = [col for col in data_df.columns]
        point_cols.remove("id")
        for col in point_cols:
            data_df[col + "_scaled"] = (data_df[col] - data_df[col].expanding().median()).astype("float32")

    def __call__(self, datasets: EegDataSet):
        for subject in datasets.subjects:
            for eeg_series in subject:
                self.transform(eeg_series.data_df)
        return datasets


class VGG16Transformers:
    SIZE = 224

    def __init__(self):
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transforms = transforms.Compose(transform_list)

    def __call__(self, *args, **kwargs):
        return self.transforms(*args, **kwargs)
