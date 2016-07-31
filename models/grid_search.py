from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import make_scorer
import pandas as pd
import function from python_file


df = pd.read_pickle('data/df.pickle')
bayes_mod = get_probs(BernoulliNB(alpha=.01), Xm, ym)
df['bayes_mod'] = bayes_mod

included_cols = ['body_length', 'channels', 'fb_published', 'has_analytics','has_logo', 'name_length', 'num_order', 'num_payouts', 'object_id', 'sale_duration2', 'show_map', 'user_age','payout_type_n', 'currency_n', 'user_type', 'bayes_mod']


y = df['fraud'].values
X = df[included_cols].values
X_train, X_test, y_train, y_test = train_test_split(X, y)


param_grid = {'learning_rate': [0.1, 0.15, .125],
              'max_depth': [8, 10, 12, 15],
              'min_samples_leaf': [3, 5, 9, 17],}
scorer = make_scorer(metrics.recall_score)
gbc = GradientBoostingClassifier(n_estimators=2000, max_depth=3)
clf = GridSearchCV(gbc, param_grid, scoring=scorer)
clf.fit(X_train, y_train)
print 'params:', clf.best_params_
print 'recall:', clf.best_score_


''''
In [42]: run grid_search.py
params: {'learning_rate': 0.1, 'max_depth': 10, 'min_samples_leaf': 3}
recall: 0.748983739837

In [43]: run grid_search.py
params: {'learning_rate': 0.2, 'max_depth': 15, 'min_samples_leaf': 9}
recall: 0.773958333333

In [44]: run grid_search.py
params: {'learning_rate': 0.1, 'max_depth': 10, 'min_samples_leaf': 17}
recall: 0.806418219462

In [46]: run grid_search.py
params: {'learning_rate': 0.15, 'max_depth': 12, 'min_samples_leaf': 9}
recall: 0.798313472442

In [47]: run grid_search.py
params: {'learning_rate': 0.125, 'max_depth': 8, 'min_samples_leaf': 5}
recall: 0.808241543296


Training:
GradientBoostingClassifier(init=None, learning_rate=0.15, loss='deviance',
              max_depth=6, max_features='auto', max_leaf_nodes=None,
              min_samples_leaf=3, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=3000,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
train time: 11.087s
test time: 0.055s
accuracy:   0.972
recall: 0.825
precision: 0.873

Out[45]:
('GradientBoostingClassifier',
 0.97182705718270568,
 0.82507288629737607,
 0.87345679012345678,
 11.08710789680481,
 0.055123090744018555)


Training:
GradientBoostingClassifier(init=None, learning_rate=0.125, loss='deviance',
              max_depth=8, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=8, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
train time: 8.353s
test time: 0.038s
accuracy:   0.973
recall: 0.816
precision: 0.892

Out[27]:
('GradientBoostingClassifier',
 0.97294281729428178,
 0.81632653061224492,
 0.89171974522292996,
 8.353060007095337,
 0.03820204734802246)

raining:
GradientBoostingClassifier(init=None, learning_rate=0.15, loss='deviance',
              max_depth=8, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=3, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=3000,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
train time: 9.224s
test time: 0.042s
accuracy:   0.972
recall: 0.816
precision: 0.886

Out[39]:
('GradientBoostingClassifier',
 0.97238493723849373,
 0.81632653061224492,
 0.88607594936708856,
 9.22350001335144,
 0.04161190986633301)
'''
