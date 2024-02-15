import os.path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.linear_model as LM
import itertools as it
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold

from pybeataml.data import AMLData
from datetime import date

# point of access for all data
data = AMLData()

en_model = LM.ElasticNet(
    random_state=0,
    max_iter=100000,
    fit_intercept=True,
    l1_ratio=.7,
    alpha=.9,
)


def run_sklearn(x_train, y_train, x_test, y_test, model, model_name):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    coef = np.array(model.coef_).flatten()
    selected_feat = model.feature_names_in_[coef > 0]
    error, r2, pearson, spearman, pr, sr = score_all(y_test, preds)

    return {
        'model': model_name,
        'feature_names': sorted(selected_feat),
        'n_feat': len(selected_feat),
        'rmse': error,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
        'pr': pr,
        'sr': sr,
    }


good_param = dict(
    device_type='cpu',
    boosting_type='gbdt',
    num_threads=8,
    n_jobs=None,
    objective='regression',
    metric='rmse',
    lambda_l1=1000,
    lambda_l2=1,
    reg_alpha=None,
    reg_lambda=None,
    learning_rate=.05,
    tree_learner='serial',
    max_bin=128,
    num_leaves=5,
    max_depth=-1,

    feature_fraction=1,  # .8

    bagging_freq=1,
    bagging_fraction=.8,
    subsample=None,
    subsample_freq=None,

    min_child_weight=0.2,
    min_data_in_leaf=2,
    min_child_samples=None,
    min_gain_to_split=None,
    colsample_bytree=None,
    min_split_gain=None,
    n_estimators=10000,
    verbose=-1,
    deterministic=True,
    random_state=10,

)


def run_gbt(x_train, y_train, x_test, y_test, feature_names):
    eval_metric = 'rmse'
    model_name = 'gbt'

    lgb_model = lgb.LGBMRegressor(**good_param)
    lgb_model.fit(
        x_train, y_train,
        callbacks=[lgb.early_stopping(
            500, first_metric_only=True, verbose=False)
        ],
        eval_metric=eval_metric,
        eval_set=[(x_train, y_train), (x_test, y_test)],
    )

    feats = pd.Series(lgb_model.feature_importances_, index=feature_names)
    selected_feat = feats[feats > 0].index.values

    preds = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration_)
    error, r2, pearson, spearman, pr, sr = score_all(y_test, preds)

    return {
        'model': model_name,
        'feature_names': sorted(selected_feat),
        'n_feat': len(selected_feat),
        'rmse': error,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
        'pr': pr,
        'sr': sr,
    }


def score_all(y_test, preds):
    error = np.sqrt(metrics.mean_squared_error(y_test, preds))
    r2 = metrics.r2_score(y_test, preds)
    pearson, pr = pearsonr(y_test, preds)
    spearman, sr = spearmanr(y_test, preds)
    return error, r2, pearson, spearman, pr, sr


def run_model(d_sets, drug_name):
    df_subset = data.get_trainable_data(d_sets, drug_name, new_format=True)
    features = df_subset.features
    target = df_subset.target
    feature_names = list(set(features.columns.values))
    # need at least 10 samples to keep test more than 2 samples
    if target.shape[0] < 10:
        return pd.DataFrame([])

    all_results = []

    kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=101)

    for n, (train_index, test_index) in enumerate(kf.split(features)):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]
        if y_test.shape[0] < 2:
            pass

        args = dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )

        gbt_results = run_gbt(feature_names=feature_names, **args)

        # enet_results = run_sklearn(
        #     model=en_model, model_name='EN', **args
        # )

        results = pd.DataFrame(
            #[enet_results, gbt_results]
            [gbt_results]
        )

        results['k'] = n
        all_results.append(results)

    if not isinstance(d_sets, str):
        out_name = '_'.join(sorted(d_sets))
    else:
        out_name = d_sets
    all_results = pd.concat(all_results)
    all_results['drug_name'] = drug_name
    all_results['data_type'] = out_name
    cols = ['model', 'n_feat', 'rmse', 'r2',
            'pearson', 'spearman', ]

    print('\t', d_sets, drug_name)
    with pd.option_context("display.precision", 2):
        print(all_results[cols].head(3))
    all_results.feature_names = all_results.feature_names.str.join('|')
    return all_results

all_sources = [
    'proteomics', 'rna_seq', 'phospho', 'acetyl',
    # 'metabolomics_HILIC', 'metabolomics_RP', 'lipidomics',  #'wes',
               ]
def run_all_sources(my_drug,
                    sources=all_sources,
                    ):
    drug_output_f_name = f'results/{my_drug}.csv'
    if os.path.exists(drug_output_f_name):
        return pd.read_csv(drug_output_f_name, index_col=0)

    # generate all possible combinations of input data
    data_sources = []
    for l in range(len(sources) + 1):
        for subset in it.combinations(sources, l):
            data_sources.append(subset)
    data_sources = data_sources[1:]  # 63

    models = []
    print(f"Working on {my_drug}")
    for j in data_sources:
        if not isinstance(j, str):
            out_name = '_'.join(sorted(j))
        else:
            out_name = j
        f_output_name = f"results/{my_drug}_{out_name}.csv"
        if not os.path.exists(f_output_name):
            data_results = run_model(j, my_drug)
            data_results.to_csv(f_output_name)
        else:
            data_results = pd.read_csv(f_output_name, index_col=0)
            print(f"Skipping {f_output_name}")
        models.append(data_results)
    results = pd.concat(models)
    results.to_csv(drug_output_f_name)
    return results


if __name__ == '__main__':
    new_data_samples = set(data.metabolomics_RP.sample_id.values)
    print(len(new_data_samples))
    new_data_samples = set(data.metabolomics_HILIC.sample_id.values)
    print(len(new_data_samples))
    new_data_samples = set(data.lipidomics.sample_id.values)
    print(len(new_data_samples))
    new_data_samples = set(data.acetyl.sample_id.values)
    print(len(new_data_samples))

    print(len(new_data_samples.intersection(data.acetyl.sample_id.values)))
    # samples don't all have rnaseq (why o why ohsu)
    print(len(new_data_samples.intersection(data.rna.sample_id.values)))
    # this allows a better comparison between datasets. New data is a subset of old data.
    data.flat_data = data.flat_data[data.flat_data.sample_id.isin(new_data_samples)].copy()
    table = data.auc_table[data.drug_names].copy()
    table = table.loc[new_data_samples, :]

    # at least 100 samples
    drug_counts = table.describe().T['count']
    high_occ_drugs = drug_counts[drug_counts > 20].index.values

    counts = table[table < 100].count()
    responsive_drugs = counts[counts > 5].index.values
    # only run single drugs for now
    drug_solo = [i for i in responsive_drugs if ' - ' not in i]
    drugs_to_focus = [
        # 'Gilteritinib', only 6 samples with data
        'Quizartinib (AC220)',
        'Trametinib (GSK1120212)',
        'Sorafenib',
        'Panobinostat',
        'Venetoclax',
    ]
    run_all_sources('Venetoclax')
    run_all_sources('Panobinostat')
    quit()
    good_drugs = set(drug_solo).intersection(high_occ_drugs)
    print(len(good_drugs))

    for i in drugs_to_focus:
        if i not in good_drugs:
            good_drugs.add(i)
    good_drugs = list(sorted(good_drugs))
    new_models = []
    for i in list(reversed(good_drugs))[10:]:
        new_models.append(run_all_sources(i))
    df = pd.concat(new_models, )

    f_name = f"regression_all_models_all_data_combos_cv_5v5_{str(date.today())}.csv"
    df.to_csv(f_name)