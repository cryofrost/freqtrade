# from tpot import TPOTRegressor
        # import sys
        # #my_tpot = TPOTRegressor(generations=10, verbosity=2, n_jobs=-2, config_dict='TPOT light')
        # my_tpot = TPOTRegressor(generations=100, verbosity=2, population_size=10, n_jobs=-2)
        # my_tpot.fit(training_features, training_target)
        # print(my_tpot.score(testing_features, testing_target))
        # my_tpot.export('tpot_exported_pipeline_news_full.py')
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
        # from sklearn.ensemble import GradientBoostingRegressor
        # from sklearn.kernel_approximation import RBFSampler
        # # from sklearn.model_selection import train_test_split
        # from sklearn.pipeline import make_pipeline
        # model = make_pipeline(
        #     RBFSampler(gamma=0.45),
        #     GradientBoostingRegressor(alpha=0.99, learning_rate=0.001, loss="lad", max_depth=10, max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=8, n_estimators=100, subsample=0.55)
        # )
        # from sklearn.neighbors import KNeighborsRegressor