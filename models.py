# import random forest
from sklearn.ensemble import RandomForestRegressor
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
import pickle


class HarmonizationModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=0):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self, X, y):
        return self.model.feature_importances_

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

    def save_model(self, path):
        pickle.dump(self.model, open(path, "wb"))



class HarmonizationUQModel:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        random_state=0,
        Dist="Normal",
        Score="CPRScore",
    ):
        self.model = NGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            Dist=Dist,
            Score=Score,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predictUQ(self, X):
        return self.model.pred_dist(X)

    def get_feature_importance(self, X, y):

        # Feature importance for pt estimation
        feature_importance_loc = ngb.feature_importances_[0]

        # Feature importance for UQ
        feature_importance_scale = ngb.feature_importances_[1]
        
        return zip(feature_importance_loc, feature_importance_scale)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)

    def save_model(self, path):
        pickle.dump(self.model, open(path, "wb"))




