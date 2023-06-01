from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import pickle

class RandomForestClassifierGridSearch:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = RandomForestClassifier(random_state=47, class_weight={0:1, 1:5})
        self.grid_search = None
        self.best_params = None
        self.best_model = None
        self.stratified_kfold = StratifiedKFold(n_splits=4,
                                                shuffle=True,
                                                random_state=47)

    def fit(self, param_grid):
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=self.stratified_kfold,
            n_jobs=-1
        )
        self.grid_search.fit(self.X, self.y)
        self.best_params = self.grid_search.best_params_
        self.best_model = self.grid_search.best_estimator_

    def get_best_params(self):
        return self.best_params

    def get_best_model(self):
        return self.best_model

    def evaluate(self, X_test, y_test):
        y_pred = self.best_model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report
    
    def save_model(self, model_name):
        with open(f'../ml_dev/models/{model_name}.pickle', 'wb') as f:
            pickle.dump(self.best_model, f)
