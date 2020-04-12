#%%
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.metrics import get_scorer
from pyearth import Earth
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import make_union
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

from pollution.linear_model import TweedieGLM
from pollution.mca import MCA

hv.extension('bokeh')


# %%
train = context.catalog.load('train').drop(columns=['Place_ID'])
test = context.catalog.load('test').drop(columns=['Place_ID'])
sample_submission = context.catalog.load('samplesubmission')

#%%
min_date = train.Date.min()

train = train.assign(Date = lambda df: (df.Date - min_date).dt.days.astype(np.float))
test = test.assign(Date = lambda df: (df.Date - min_date).dt.days.astype(np.float))

#%%
train.target.hvplot.kde()


# %%
transformer = Pipeline([('inpute', SimpleImputer()),#IterativeImputer(random_state=0)),
                        ('poly', PolynomialFeatures()),
                        ('scale', StandardScaler()),
                        ('pca', PCA(15)),
                        ('rescale', StandardScaler())])

glm = TweedieGLM(power=0, max_iter=1000)
mars = Earth()
gb = HistGradientBoostingRegressor()
estimators = [
    ('mars', mars),
    ('gb', gb)
]

final_estimator = Pipeline([
                        ('poly', PolynomialFeatures(2)),
                        ('scale', StandardScaler()),
                        ('pca', PCA()),
                        ('regressor', LinearRegression())])

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator
)

model = Pipeline([('transformer', transformer),
                  ('model', stack)])

offset = 1e-9
def add(y):
    return np.log(y + offset)

def subtract(y):
    return (np.exp(y) - offset)


link = Pipeline([('function', FunctionTransformer(add, subtract, validate=True))])
scorer = get_scorer('neg_root_mean_squared_error')

pipeline = TransformedTargetRegressor(regressor=model, transformer=link)



#%%
glm_params = {'regressor__model__power': [0, 2, 3],
          'regressor__model__alpha': [1e-3, 1e-1, 1],
          'regressor__transformer__pca__n_components': [10, 25, 45]}

mars_params = {'regressor__model__mars__max_degree': [1, 2],
          'regressor__model__mars__max_terms': [15, 20],
          'regressor__transformer__pca__n_components': [10, 25],
          'regressor__transformer__poly__degree': [1]}

search = RandomizedSearchCV(pipeline, mars_params, scoring=scorer, 
n_iter = 1, n_jobs=-1, return_train_score=True)

X_train, y_train = train.drop(columns=['target', 'target_min', 'target_max', 'target_variance']), train.target

search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_)
context.io.save('searchresults', results)


# %%
X_test = test.loc[:, X_train.columns]

# %%
# predict and plot
y_pred = pd.Series(search.predict(X_test), name = y_train.name).clip(0)
submissionkde = y_pred.hvplot.kde()


# %%
# format submission
submission = sample_submission
submission.target = y_pred

context.io.save('submission', submission)

# %%
results.sort_values('mean_test_score').tail()
