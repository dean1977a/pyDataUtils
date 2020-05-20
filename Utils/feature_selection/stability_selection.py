'''
特征稳定性
https://thuijskens.github.io/2018/07/25/stability-selection/
https://zhuanlan.zhihu.com/p/110643632
'''

import numpy as np


def stability_selection(lasso, alphas, n_bootstrap_iterations,
                        X, y, seed):
  n_samples, n_variables = X.shape
  n_alphas = alphas.shape[0]

  rnd = np.random.RandomState(seed)
  selected_variables = np.zeros((n_variables,
                                 n_bootstrap_iterations))
  stability_scores = np.zeros((n_variables, n_alphas))

  for idx, alpha, in enumerate(alphas):
    # This is the sampling step, where bootstrap samples are generated
    # and the structure learner is fitted
    for iteration in range(n_bootstrap_iterations):
      bootstrap = rnd.choice(np.arange(n_samples),
                             size=n_samples // 2,
                             replace=False)

      X_train = X[bootstrap, :]
      y_train = y[bootstrap]

      # Assume scikit-learn implementation
      lasso.set_params({'C': alpha}).fit(X_train, y_train)
      selected_variables[:, iteration] = (np.abs(lasso.coef_) > 1e-4)

    # This is the scoring step, where the final stability
    # scores are computed
    stability_scores[:, idx] = selected_variables.mean(axis=1)

  return stability_scores

base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1'))
])

selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',
                              lambda_grid=np.logspace(-5, -1, 50)).fit(X, y)

print(selector.get_support(indices=True))