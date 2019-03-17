from unittest import TestCase

from utils.data.eegdataloader import EegDataLoader


class TestEegSeries(TestCase):

    def test_crop(self):
        dataset = EegDataLoader(is_debug=True)()

        cropped = dataset.subjects[0][0].crop(10000, 5000, release_raw=True)
        print(cropped)
        # TODO assert
