
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

# Perform hyperparameter tuning for the model
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    model = HistGradientBoostingClassifier()
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3)
    search.fit(X_train, y_train)
    return search.best_params_
