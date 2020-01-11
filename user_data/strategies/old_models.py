
# from tpot import TPOTRegressor
# import sys
# #my_tpot = TPOTRegressor(generations=10, verbosity=2, n_jobs=-2, config_dict='TPOT light')
# my_tpot = TPOTRegressor(generations=100, verbosity=2, population_size=10, n_jobs=-2)
# my_tpot.fit(training_features, training_target)
# print(my_tpot.score(testing_features, testing_target))
# my_tpot.export('tpot_exported_pipeline_news_full.py')
# sys.exit(0)

# import keras
#         import sys
#         import tensorflow as tf
#         from keras.models import Model
#         from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, Flatten
#         from keras import optimizers
#         # import numpy as np
#         numpy.random.seed(4)
#         # from tensorflow import set_random_seed
#         # set_random_seed(4)
        
#         ishape=training_features[0].shape
#         # ashape=training_features.shape
#         n_n = ishape[1]
#         # ValueError: Error when checking input: expected lstm_input to have shape (120, 1) but got array with shape (120, 6)

#         lstm_input = Input(shape=ishape, name='lstm_input')
#         x = LSTM(10 * n_n, name='lstm_0')(lstm_input)
#         x = Dropout(0.2, name='lstm_dropout_0')(x)
#         # x = Flatten()(x)
#         # x = Dense(2 ** (n_n + 1), name='dense_0')(x)
#         x = Dense(12 * n_n, name='dense_0')(x)
#         x = Activation('sigmoid', name='sigmoid_0')(x)
#         x = Dense(1, name='dense_1')(x)
#         output = Activation('linear', name='linear_output')(x)
#         model = Model(inputs=lstm_input, outputs=output)

#         adam = optimizers.Adam(lr=0.0005)

#         model.compile(optimizer=adam, loss='mse')

# model.fit(x=training_features, y=training_target, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
        # model.save('./user_data/model/lstm.mdl')
                # from keras.utils import plot_model
        # plot_model(model, to_file='./user_data/model/model.png', show_shapes=True, expand_nested=True)

        # sys.exit(0)
        # model = keras.models.load_model('./user_data/model/lstm.mdl')


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
        # from sklearn.ensemble import GradientBoostingRegressor
        # from sklearn.kernel_approximation import RBFSampler
        # # from sklearn.model_selection import train_test_split
        # from sklearn.pipeline import make_pipeline
        # model = make_pipeline(
        #     RBFSampler(gamma=0.45),
        #     GradientBoostingRegressor(alpha=0.99, learning_rate=0.001, loss="lad", max_depth=10, max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=8, n_estimators=100, subsample=0.55)
        # )
        # from sklearn.neighbors import KNeighborsRegressor