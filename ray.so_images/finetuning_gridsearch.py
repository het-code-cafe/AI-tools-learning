"""
finetuning_gridsearch.py

From: https://www.projectpro.io/recipes/find-optimal-parameters-using-gridsearchcv#mcetoc_1g1istqfd7
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

dataset = datasets.load_wine()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

GBC = GradientBoostingClassifier()

parameters = {'learning_rate': [0.01, 0.02, 0.03],
              'subsample': [0.9, 0.5, 0.2],
              'n_estimators': [100, 500, 1000],
              'max_depth': [4, 6, 8]
              }

grid_GBC = GridSearchCV(estimator=GBC, param_grid = parameters, cv = 2, n_jobs=-1)
grid_GBC.fit(X_train, y_train)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_GBC.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_GBC.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_GBC.best_params_)