from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error


def train_tree_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators= 200,
        max_depth=15,
        random_state=42,
        n_jobs = 1
    )
    model.fit(X_train,y_train)

    return model

def evaluate_tree_model(model, X, y):
    predictions = model.predict(X)
    mae = mean_absolute_error(y,predictions)
    rmse = root_mean_squared_error(y,predictions)
    
    return mae,rmse
