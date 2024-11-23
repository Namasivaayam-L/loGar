import torch
from sklearn.model_selection import GridSearchCV

# Define your model and training loop

param_grid = {'batch_size': [8, 16, 32, 64, 128]}
grid_search = GridSearchCV(estimator=your_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

best_batch_size = grid_search.best_params_['batch_size']