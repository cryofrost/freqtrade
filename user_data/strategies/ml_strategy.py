"""
Strategy based on machine learning algorythms
"""
# --- Do not remove these libs ---
# import freqtrade.vendor.qtpylib.indicators as qtpylib
import glob
import logging
import time
import warnings
# from typing import Dict, List
# from functools import reduce
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.externals import joblib
# from sklearn.preprocessing import Imputer

# from sklearn.pipeline import make_pipeline, make_union
# from sklearn.preprocessing import Binarizer, MinMaxScaler
# from sklearn.tree import DecisionTreeRegressor
# from tpot.builtins import StackingEstimator, ZeroCount

# --------------------------------

from datetime import datetime
from datetime import timezone
from time import mktime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import nltk
import numpy # noqa
import feedparser
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger('ML Strategy')
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', download_dir='./user_data/nltk_data/')


class MLStrategy(IStrategy):
    """
    Strategy 005
    author@: Gerald Lonlas
    github@: https://github.com/freqtrade/freqtrade-strategies

    How to use it?
    > python3 ./freqtrade/main.py -s Strategy005
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        # "1440": 0.01,
        "80": 0.02,
        "40": 0.03,
        "20": 0.04,
        "0":  0.05
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.90

    # Optimal ticker interval for the strategy
    ticker_interval = '1m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    ta_on_candle = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def __init__(self, config: dict) -> None:
        print("init")
        # print(config)
        super().__init__(config)
        self.is_trained = False
        # if not self.is_trained:
        if not hasattr(self, 'dp'):
            from freqtrade.resolvers import ExchangeResolver
            from freqtrade.data.dataprovider import DataProvider

            exchange = ExchangeResolver(
                self.config['exchange']['name'], self.config, False).exchange
            dataprovider = DataProvider(self.config, exchange)
            self.dp = dataprovider

        self.train()

    def train(self):
        """
        Trains a model
        """
        start = time.time()
        response = "Train failure"

        if self.dp:
            pair = self.dp._config['pairs'][0]
            dataframe = self.dp.historic_ohlcv(pair)
            metadata = {'pair': pair}
            dataframe = self.populate_indicators(dataframe, metadata)

        df = dataframe.dropna()
        # print(df.keys())
        df.set_index('date', drop=False)
        # print(df.tail())
        self.to_drop = ['date',
                        # 'volume',
                        # 'close_to_sma',
                        # 'high', 'low',
                        # 'open', 'close',
                        # 'macd',
                        'macdsignal',
                        # 'minus_di',
                        # 'TRANGE'
                        # 'rsi',
                        'fisher_rsi', 'fisher_rsi_norma',
                        # 'fastd', 'fastk',
                        # 'sar', 'sma'
                        ]

        df_filtered = df.drop(self.to_drop, axis=1)
        self.fr_features = ['high', 'low', 'open', 'close',
                            'volume', 'minus_di', 'rsi', 'fastd', 'fastk',
                            'sar', 'sma', 'TRANGE']
        self.neg_features = ['macd']
        # df[['x','z']] = mms.fit_transform(df[['x','z']])
        self.fr_scaler = MinMaxScaler(feature_range=(0, 1))
        df_filtered[self.fr_features] = self.fr_scaler.fit_transform(df_filtered[self.fr_features])
        self.neg_scaler = MinMaxScaler(feature_range=(-1, 1))
        df_filtered[self.neg_features] = self.neg_scaler.fit_transform(df_filtered[self.neg_features])
        features = (pd.np.array(df_filtered.values))

        future = -1  # to shift forward
        label = df.shift(future).copy()
        target = label['perc_change'].dropna().values
        # label = label['close'].values.reshape(-1, 1)
        # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        # label = imp.fit_transform(label)
        # label[0] = label[1]
        training_features, testing_features, training_target, testing_target = \
            train_test_split(features[:future], target, random_state=0, shuffle=False)

        for i in range(1):
            print('train: ', training_features[i], training_target[i])
        #     print('test:', testing_features[i], testing_target[i])
        # from tpot import TPOTRegressor
        # import sys
        # #my_tpot = TPOTRegressor(generations=10, verbosity=2, n_jobs=-2, config_dict='TPOT light')
        # my_tpot = TPOTRegressor(generations=10, verbosity=2, n_jobs=-2)
        # my_tpot.fit(training_features, training_target)
        # print(my_tpot.score(testing_features, testing_target))
        # my_tpot.export('tpot_exported_pipeline_scaled_full.py')
        # sys.exit(0)

        # from sklearn.ensemble import GradientBoostingRegressor
        # model = GradientBoostingRegressor(alpha=0.9,
        #                                   learning_rate=0.9,
        #                                   loss="huber",
        #                                   max_depth=8,
        #                                 #   max_features=0.8500000000000001,
        #                                 #   min_samples_leaf=1,
        #                                 #   min_samples_split=14,
        #                                   n_estimators=20,
        #                                 #   subsample=0.55
        #                                   )
        # import xgboost as xgb
        # model = xgb.XGBRegressor(alpha=0.85,
        #                          learning_rate=0.9,
        #                          loss="huber",
        #                          max_depth=8,
        #                     #   max_features=0.8500000000000001,
        #                     #   min_samples_leaf=1,
        #                     #   min_samples_split=14,
        #                          n_estimators=10,
        #                     #   subsample=0.55
        #                          )

        # from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=3, p=2, weights="distance", n_jobs=4)
        model.fit(training_features, training_target)
        response = 'Model fitted using dataframe of length = '\
                    + str(len(training_features))\
                    + ' during '\
                    + str(time.time() - start)\
                    + ' seconds with testing score = '
        print(response, model.score(testing_features, testing_target))
        # joblib.dump(model, 'gdb_' + pair + '.pkl')
        self.model = model
        self.is_trained = True


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def analyze_news(self, dataframe: DataFrame):
        """
        Download, parse and do sentiment analysis
        """
        entries = []
        sia = SentimentIntensityAnalyzer()
        # # stock market lexicon
        stock_lex = pd.read_csv('./user_data/lexicon_data/stock_lex.csv')
        stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
        stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))
        stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
        stock_lex_scaled = {}
        for k, v in stock_lex.items():
            if v > 0:
                stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
            else:
                stock_lex_scaled[k] = v / min(stock_lex.values()) * -4

        final_lex = {}
        final_lex.update(stock_lex_scaled)
        final_lex.update(sia.lexicon)
        sia.lexicon = final_lex

        if not self.is_trained:
            for filename in glob.iglob("./user_data/news/" + '**/*.html', recursive=True):
                logger.info('parsing %s', filename)
                nf = feedparser.parse(filename)
                entries += nf.entries
            self.training_news = entries
        else:
            if self.dp.runmode.value in ('backtest'):
                entries = self.training_news
            else:
                feed_url = "https://news.bitcoin.com/feed/"
                logger.info('parsing %s', feed_url)
                nf = feedparser.parse(feed_url)
                entries = nf.entries


        dataframe = dataframe.set_index("date", drop=False)
        col = 'sentiment'
        dataframe = dataframe.assign(**{col:numpy.full(len(dataframe), numpy.nan)})

        for entry in entries:
            html = entry.content[0]['value']
            link_soup = BeautifulSoup(html)
            sentences = link_soup.findAll("p")
            passage = ""
            for sentence in sentences:
                passage += sentence.text
           
            sentiment = sia.polarity_scores(passage)['compound']
            dt = datetime.fromtimestamp(mktime(entry['published_parsed']))
            pd_dt = pd.Timestamp(dt.replace(tzinfo=timezone.utc), freq='T')
            idx = dataframe.index.get_loc(pd_dt, method='nearest')
            candle_date = dataframe.iloc[idx].date
            dataframe.loc[(dataframe.date == candle_date), col] = sentiment

            logger.info("Got alfa {} for candle at {}".format(sentiment, candle_date))

        # from sklearn.impute import SimpleImputer
        # imp = SimpleImputer(missing_values='NaN', strategy='mean')
        # sentiments = pd.np.array(dataframe.sentiment).reshape(-1, 1)
        dataframe = dataframe.fillna(method='ffill')
        dataframe = dataframe.fillna(method='bfill')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # Minus Directional Indicator / Movement
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # Overlap Studies
        # ------------------------------------

        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        # dataframe['close_to_sma'] = dataframe['close'] / dataframe['sma']

        dataframe['TRANGE'] = ta.TRANGE(dataframe)
        dataframe['perc_change'] = 100 * (dataframe['close'] - dataframe['open']) / dataframe['open']

        # News feed analysis
        dataframe = self.analyze_news(dataframe)
        # dataframe['alfa'] = alfa['sentiment']

        if self.is_trained:
            dataframe = dataframe.dropna()
            dataframe['future_perc_change'] = self.predict(dataframe)

        return dataframe

    def predict(self, dataframe: DataFrame):
        df = dataframe
        df_filtered = df.drop(self.to_drop, axis=1)
        df_filtered[self.fr_features] = self.fr_scaler.fit_transform(df_filtered[self.fr_features])
        df_filtered[self.neg_features] = self.neg_scaler.fit_transform(df_filtered[self.neg_features])
        features = (pd.np.array(df_filtered.values))
        # features = pd.np.array(df.drop(self.to_drop, axis=1).values)
        predcs_class = self.model.predict(features)
        for i in range(1):
            print('predict: ', features[i], predcs_class[i])
        return predcs_class

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # Prod
            # (
            #     (dataframe['close'] > 0.00000200) &
            #     (dataframe['volume'] > dataframe['volume'].mean() * 4) &
            #     (dataframe['close'] < dataframe['sma']) &
            #     (dataframe['fastd'] > dataframe['fastk']) &
            #     (dataframe['rsi'] > 0) &
            #     (dataframe['fastd'] > 0) &
            #     # (dataframe['fisher_rsi'] < -0.94)
            #     (dataframe['fisher_rsi_norma'] < 38.900000000000006) &
            #     # (dataframe['future_close'] > dataframe['close'])
            # ),
            (dataframe['future_perc_change'] > dataframe['perc_change']),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # Prod
            # (
            #     (qtpylib.crossed_above(dataframe['rsi'], 50)) &
            #     (dataframe['macd'] < 0) &
            #     (dataframe['minus_di'] > 0)
            # ) |
            # (
            #     (dataframe['sar'] > dataframe['close']) &
            #     dataframe['fisher_rsi'] > 0.3
            # ),
            (dataframe['future_perc_change'] < dataframe['perc_change']),
            'sell'] = 0
        return dataframe
