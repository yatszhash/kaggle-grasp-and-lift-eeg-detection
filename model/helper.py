import logging

logger = logging.getLogger(__name__)


class StudyFactory(object):

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def __call__(self, *args, **kwargs):
        pass

#
# class():
#
# class Predictor(object):
#
#     def __init__(self, linker, model, features, market_preprocess, news_preprocess):
#         self.linker = linker
#         self.model = model
#         self.features: Features = features
#         self.market_preprocess = market_preprocess
#         self.news_preprocess = news_preprocess
#
#     def predict_all(self, days, env):
#         logger.info("=================prediction start ===============")
#
#         stored_market_df = None
#         stored_news_df = None
#         max_time = None
#         predict_start_id = 0
#
#         def store_past_data(market_df, news_df, max_store_date=0):
#             nonlocal stored_market_df
#             nonlocal stored_news_df
#             nonlocal predict_start_id
#             if stored_market_df is None or max_store_date == 0:
#                 stored_market_df = market_df
#                 stored_news_df = news_df
#                 predict_start_id = 0
#                 return
#
#             nonlocal max_time
#             max_time = market_df["time"].max()
#
#             min_time = max_time - offsets.Day(max_store_date)
#             stored_market_df = stored_market_df[stored_market_df["time"] >= min_time]
#             stored_news_df = stored_news_df[stored_news_df["firstCreated"] >= min_time]
#
#             predict_start_id = len(stored_market_df)
#
#             stored_market_df = pd.concat([stored_market_df, market_df], axis=0, ignore_index=True)
#             stored_news_df = pd.concat([stored_news_df, news_df], axis=0, ignore_index=True)
#
#         for (market_obs_df, news_obs_df, predictions_template_df) in tqdm(days):
#             store_past_data(market_obs_df, news_obs_df, FeatureSetting.max_shift_date)
#             market_obs_df_cp, news_obs_df_cp = stored_market_df.copy(), stored_news_df.copy()
#             self.make_predictions(market_obs_df_cp, news_obs_df_cp, predictions_template_df, predict_start_id)
#             env.predict(predictions_template_df)
#
#     def make_predictions(self, market_obs_df, news_obs_df, predictions_df, predict_id_start):
#         logger.info("predicting....")
#
#         market_obs_df = self.market_preprocess.transform(market_obs_df)
#         news_obs_df = self.news_preprocess.transform(news_obs_df)
#         # print(market_obs_df[MARKET_ID])
#
#         market_obs_df, news_obs_df = self.features.transform(market_obs_df, news_obs_df)
#
#         if FeatureSetting.should_use_news_feature:
#             self.linker.link(market_obs_df, news_obs_df)
#             market_obs_df = self.linker.create_new_market_df()
#             self.linker.clear()
#             # print(market_obs_df[MARKET_ID])
#         del news_obs_df
#         gc.collect()
#
#         feature_matrix = self.features.get_linked_feature_matrix(market_obs_df)
#
#         logger.info("input size: {}".format(feature_matrix.shape))
#         predictions = self.model.predict(feature_matrix)
#         predict_indices = market_obs_df[MARKET_ID][market_obs_df[MARKET_ID] >= predict_id_start].astype("int").tolist()
#
#         logger.info("predicted size: {}".format(predictions.shape))
#         # logger.info("predicted indices: {}".format(predict_indices))
#         # print(predict_indices)
#         predictions = predictions[predict_indices] * 2 - 1
#         predictions = predictions[np.argsort(predict_indices)]
#         logger.info("predicted size: {}".format(predictions.shape))
#         logger.info("predicted target size: {}".format(predictions_df.shape))
#         predictions_df.confidenceValue = predictions
#         logger.info("prediction done")
