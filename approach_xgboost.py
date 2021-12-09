"""
    File name: approach_xgboost.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""
import os
import sys
from statistics import mean

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn import tree

from matplotlib import pyplot as plt
from scipy import stats
from sklearn import linear_model

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

import utils
from loader_representations import load_rep, load_rep_info

from collections import Counter

from xgboost import XGBClassifier
from scipy.stats import pearsonr, spearmanr, kendalltau


def get_split(random_state, X, paths):
    split = utils.load_json("mikrokosmos/splits.json")[str(random_state)]
    X_train, X_test, y_train, y_test, ids_train, ids_test = [], [], [], [], [], []
    # train
    for y, idx in zip(split['y_train'], split['ids_train']):
        index = np.where(paths == idx)[0][0]
        X_train.append(X[index])
        y_train.append(y)
        ids_train.append(int(os.path.basename(idx)[:-4]))

    # test
    for y, idx in zip(split['y_test'], split['ids_test']):
        index = np.where(paths == idx)[0][0]
        X_test.append(X[index])
        y_test.append(y)
        ids_test.append(int(os.path.basename(idx)[:-4]))
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), np.array(ids_train), np.array(
        ids_test)


def plot_results(y_pred_train, y_train, y_pred_test, y_test, path):
    cm = confusion_matrix(y_train, y_pred_train, labels=range(3))
    df = pd.DataFrame(cm, index=range(3), columns=range(3))
    plt.figure(figsize=(7, 7))
    sns.heatmap(df, annot=True, cbar=False)
    plt.title('Confusion matrix for train set predictions', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(path)
    # plt.show()

    cm = confusion_matrix(y_test, y_pred_test, labels=range(3))
    df = pd.DataFrame(cm, index=range(3), columns=range(3))
    plt.figure(figsize=(7, 7))
    sns.heatmap(df, annot=True, cbar=False)
    plt.title('Confusion matrix for test set predictions', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(path)
    # plt.show()


def create_dataset(X, y, ids, win_size=9):
    X_ans, y_ans, id_tracks = [], [], []
    hop_size = 1
    for x, label, id_track in zip(X, y, ids):
        for i in range(1, x.shape[0] - win_size + 1, hop_size):
            window = x[i:i + win_size, :].reshape((-1, 1))  # each individual window
            X_ans.append(window)
            y_ans.append(label)
            id_tracks.append(id_track)
    X_ans = np.squeeze(np.array(X_ans))
    return X_ans, np.array(y_ans), id_tracks


def cross_validation(y, ids):
    unique = set([(id, yy) for id, yy in zip(ids, y)])
    unique_ids, unique_y = zip(*unique)
    sfk = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for train_id_index, test_id_index in sfk.split(unique_ids, unique_y):
        train_ids = [unique_ids[jj] for jj in train_id_index]
        validation_ids = [unique_ids[jj] for jj in test_id_index]
        yield ([idx for idx, id_n in enumerate(ids) if id_n in train_ids],
               [idx for idx, id_n in enumerate(ids) if id_n in validation_ids])


# second step
def create_histogram(y_patch, y_pred, ids):
    grouped_results = {ii: [] for ii in set(ids)}
    for yy_pred, idy in zip(y_pred, ids):
        grouped_results[idy] = grouped_results[idy] + [yy_pred]

    ids2y = {idx: yy for idx, yy in zip(ids, y_patch)}
    histo = []
    y_global = []
    for idy, preds in grouped_results.items():
        element = {
            0: preds.count(0) / len(preds),
            1: preds.count(1) / len(preds),
            2: preds.count(2) / len(preds),
        }
        histo.append(element)
        y_global.append(ids2y[idy])
    histo = pd.DataFrame(histo)

    return histo.to_numpy(), y_global


def step1(split, split_number, plot=True, verbose=True, save=True, exp="XXX", window_size=9, precomputed=False,
          gpu=False):
    print(f"EXECUTING SPLIT {split_number}")
    X_train, X_test, y_train, y_test, paths_train, paths_test = split
    X_train, y_train, ids_train = create_dataset(X_train, y_train, paths_train, win_size=window_size)
    X_test, y_test, ids_test = create_dataset(X_test, y_test, paths_test, win_size=window_size)
    if not precomputed:
        total_samples = len(y_train)
        weights_classes = {k: 1 / (x / total_samples) for k, x in sorted(Counter(y_train).items())}
        weights = [weights_classes[yy] for yy in y_train]

        param_dist = {
            'n_estimators': stats.randint(150, 800),
            'learning_rate': stats.uniform(0.01, 0.59),
            'subsample': stats.uniform(0.3, 0.6),
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'colsample_bytree': stats.uniform(0.5, 0.4),
            'min_child_weight': range(1, 12, 2),
            'gamma': [i / 10.0 for i in range(0, 5)],
        }

        if gpu:
            model = XGBClassifier(n_jobs=12, tree_method='gpu_hist', gpu_id=0)  # Use GPU
        else:
            model = XGBClassifier(n_jobs=12)  # Use CPU

        fit_params = {
            "early_stopping_rounds": 25,
            "eval_metric": "mlogloss",
            "eval_set": [[X_test, y_test]],
            "sample_weight": weights
        }
        # grid = grid(model, hyperparameters, scoring='f1_weighted', cv=list(cross_validation(y_train, ids_train)), verbose=10)  # Fit the model

        grid = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            cv=list(cross_validation(y_train, ids_train)),
            n_iter=25,
            scoring='balanced_accuracy',
            error_score=0,
            verbose=3,
        )
        # redirect stout because verbose_eval does not work
        best_model = grid.fit(
            X_train,
            y_train,
            **fit_params
        )
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        if plot:
            plot_results(y_pred_train, y_train, y_pred_test, y_test,
                         f"results/img/{exp}_step1_w{window_size}_s{split_number}.png")

        train_acc = accuracy_score(y_pred=y_pred_train, y_true=y_train)
        test_acc = accuracy_score(y_pred=y_pred_test, y_true=y_test)
        train_balanced_acc = balanced_accuracy_score(y_pred=y_pred_train, y_true=y_train)
        test_balanced_acc = balanced_accuracy_score(y_pred=y_pred_test, y_true=y_test)

        if verbose:
            print("STEP 1")
            print("Best parameters: {}".format(grid.best_params_))
            print('Train acc', train_acc)
            print('Test acc', test_acc)
            print('Train b acc', train_balanced_acc)
            print('Test b acc', test_balanced_acc)

        if save:
            if not os.path.exists(f"results/xgboost/results_{exp}_w{window_size}.json"):
                utils.save_json({}, f"results/xgboost/results_{exp}_w{window_size}.json")
            results = utils.load_json(f"results/xgboost/results_{exp}_w{window_size}.json")

            results[str(split_number)] = {
                'train_balanced_acc': train_balanced_acc,
                'test_balanced_acc': test_balanced_acc,
                "best_parameters": f"split:{str(split_number).zfill(2)} "
                                   f"n_estimators:{grid.best_params_['n_estimators']} "
                                   f"learning_rate:{grid.best_params_['learning_rate']} "
                                   f"subsample:{grid.best_params_['subsample']} "
                                   f"max_depth:{grid.best_params_['max_depth']} "
                                   f"colsample_bytree:{grid.best_params_['colsample_bytree']} "
                                   f"gamma:{grid.best_params_['gamma']} "
                                   f"min_child_weight:{grid.best_params_['min_child_weight']} "
            }
            utils.save_json(results, f"results/xgboost/results_{exp}_w{window_size}.json")
            grid.best_estimator_.save_model(f'results/xgboost/{exp}/w{window_size}/{split_number}.pkl')
    else:
        clf = xgboost.XGBClassifier()
        clf.load_model(f'results/xgboost/{exp}/w{window_size}/{split_number}.pkl')
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_acc = accuracy_score(y_pred=y_pred_train, y_true=y_train)
        test_acc = accuracy_score(y_pred=y_pred_test, y_true=y_test)
        train_balanced_acc = balanced_accuracy_score(y_pred=y_pred_train, y_true=y_train)
        test_balanced_acc = balanced_accuracy_score(y_pred=y_pred_test, y_true=y_test)

        if verbose:
            print("STEP 1")
            print('Train acc', train_acc)
            print('Test acc', test_acc)
            print('Train b acc', train_balanced_acc)
            print('Test b acc', test_balanced_acc)

    return (y_train, y_pred_train, ids_train), (y_test, y_pred_test, ids_test)


def step2_trees(split_number, train_results, test_results, exp, plot=True, verbose=True, save=True, window_size=9):
    y_train, y_pred_train, ids_train = train_results
    y_test, y_pred_test, ids_test = test_results

    X2_train, y2_train = create_histogram(y_train, y_pred_train, ids_train)
    X2_test, y2_test = create_histogram(y_test, y_pred_test, ids_test)

    model = XGBClassifier(
        max_depth=2,
        gamma=2,
        eta=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5
    )
    model.fit(X2_train, y2_train)
    y2_pred_train = model.predict(X2_train)
    y2_pred_test = model.predict(X2_test)
    train_acc2 = accuracy_score(y_pred=y2_pred_train, y_true=y2_train)
    test_acc2 = accuracy_score(y_pred=y2_pred_test, y_true=y2_test)
    train_balanced_acc2 = balanced_accuracy_score(y_pred=y2_pred_train, y_true=y2_train)
    test_balanced_acc2 = balanced_accuracy_score(y_pred=y2_pred_test, y_true=y2_test)
    if verbose:
        print("STEP 2")
        # print("Best parameters: {}".format(grid2.best_params_))
        print('Train acc', train_acc2)
        print('Test acc', test_acc2)
        print('Train b acc', train_balanced_acc2)
        print('Test b acc', test_balanced_acc2)

    if save:
        if not os.path.exists(f"results/xgboost/step2_results_{exp}_w{window_size}.json"):
            utils.save_json({}, f"results/xgboost/step2_results_{exp}_w{window_size}.json")
        results2 = utils.load_json(f"results/xgboost/step2_results_{exp}_w{window_size}.json")
        results2[str(split_number)] = {
            'train_balanced_acc': train_balanced_acc2,
            'test_balanced_acc': test_balanced_acc2
        }
        utils.save_json(results2, f"results/xgboost/step2_results_{exp}_w{window_size}.json")
        joblib.dump(model, f'results/xgboost/{exp}/w{window_size}/step2_{split_number}.pkl')

    if plot:
        plot_results(y2_pred_train, y2_train, y2_pred_test, y2_test,
                     f"results/img/{exp}_step2_w{window_size}_s{split_number}.png")


def step2_logistic(split_number, train_results, test_results, exp, plot=True, verbose=True, save=True,
                   window_size=9):
    y_train, y_pred_train, ids_train = train_results
    y_test, y_pred_test, ids_test = test_results

    X2_train, y2_train = create_histogram(y_train, y_pred_train, ids_train)
    X2_test, y2_test = create_histogram(y_test, y_pred_test, ids_test)

    parameters = dict(C=[1],
                      penalty=['l1'],
                      class_weight=["balanced"],
                      solver=["liblinear"])

    lr = linear_model.LogisticRegression()
    grid2 = GridSearchCV(lr, parameters, scoring="balanced_accuracy")
    best_model2 = grid2.fit(X2_train, y2_train)

    y2_pred_train = best_model2.predict(X2_train)
    y2_pred_test = best_model2.predict(X2_test)
    train_acc2 = accuracy_score(y_pred=y2_pred_train, y_true=y2_train)
    test_acc2 = accuracy_score(y_pred=y2_pred_test, y_true=y2_test)
    train_balanced_acc2 = balanced_accuracy_score(y_pred=y2_pred_train, y_true=y2_train)
    test_balanced_acc2 = balanced_accuracy_score(y_pred=y2_pred_test, y_true=y2_test)
    if verbose:
        print("STEP 2")
        print("Best parameters: {}".format(grid2.best_params_))
        print('Train acc', train_acc2)
        print('Test acc', test_acc2)
        print('Train b acc', train_balanced_acc2)
        print('Test b acc', test_balanced_acc2)

    if save:
        if not os.path.exists(f"results/xgboost/step2tree_results_{exp}_w{window_size}.json"):
            utils.save_json({}, f"results/xgboost/step2tree_results_{exp}_w{window_size}.json")
        results2 = utils.load_json(f"results/xgboost/step2tree_results_{exp}_w{window_size}.json")
        results2[str(split_number)] = {
            'train_balanced_acc': train_balanced_acc2,
            'test_balanced_acc': test_balanced_acc2
        }
        utils.save_json(results2, f"results/xgboost/step2tree_results_{exp}_w{window_size}.json")
        joblib.dump(grid2, f'results/xgboost/{exp}/w{window_size}/step2tree_{split_number}.pkl')

    if plot:
        plot_results(y2_pred_train, y2_train, y2_pred_test, y2_test,
                     f"results/img/{exp}_step2tree_w{window_size}_s{split_number}.png")


def compute_mean(y_patch, y_pred, ids):
    grouped_results = {ii: [] for ii in set(ids)}
    for yy_pred, idy in zip(y_pred, ids):
        grouped_results[idy] = grouped_results[idy] + [yy_pred]

    ids2y = {idx: yy for idx, yy in zip(ids, y_patch)}
    y_global = []
    m = []
    ids_ans = []
    # import pdb; pdb.set_trace()
    for idy, preds in grouped_results.items():
        ids_ans.append(idy)
        m.append(np.mean(preds))
        y_global.append(ids2y[idy])
    # print(m, y_global)
    # import pdb;pdb.set_trace()
    return m, y_global, ids_ans


def predict_threshold(t1, t2, elements):
    ans = []
    for e in elements:
        if e < t1:
            ans.append(0)
        elif t1 <= e <= t2:
            ans.append(1)
        else:
            ans.append(2)
    return ans


# def generate_feasible_states(dimY, dimX):
#     for idx in range(dimY):


def step2_threshold(split_number, train_results, test_results, exp, plot=True, verbose=True, save=True,
                    window_size=9):
    y_train, y_pred_train, ids_train = train_results
    y_test, y_pred_test, ids_test = test_results
    # import pdb;pdb.set_trace()
    X2_train, y2_train, _ = compute_mean(y_train, y_pred_train, ids_train)
    X2_test, y2_test, _ = compute_mean(y_test, y_pred_test, ids_test)

    X_train_histogram, y_train_histogram = create_histogram(y_train, y_pred_train, ids_train)
    X_test_histogram, y_test_histogram = create_histogram(y_test, y_pred_test, ids_test)

    # import pdb; pdb.set_trace()
    # y2_train, y2_validation = list(cross_validation(y2_train, ids_train))[0]
    best = (0, 0)
    best_predict = 0
    for t1 in np.arange(0.1, 2, 0.1):
        for t2 in np.arange(t1, 1.9, 0.1):
            partial_predict = predict_threshold(t1, t2, X2_train)
            bacc = balanced_accuracy_score(y_pred=partial_predict, y_true=y2_train)
            # print(bacc, t1, t2)
            if bacc > best_predict:
                best_predict = bacc
                best = (t1, t2)
    t1, t2 = best
    y2_pred_train = predict_threshold(t1, t2, X2_train)
    y2_pred_test = predict_threshold(t1, t2, X2_test)
    train_acc2 = accuracy_score(y_pred=y2_pred_train, y_true=y2_train)
    test_acc2 = accuracy_score(y_pred=y2_pred_test, y_true=y2_test)

    train_balanced_acc2 = balanced_accuracy_score(y_pred=y2_pred_train, y_true=y2_train)
    test_balanced_acc2 = balanced_accuracy_score(y_pred=y2_pred_test, y_true=y2_test)

    print({'t1': t1, 't2': t2})
    if verbose:
        print("STEP 2")
        print('Train acc', train_acc2)
        print('Test acc', test_acc2)
        print('Train b acc', train_balanced_acc2)
        print('Test b acc', test_balanced_acc2)

    if save:
        if not os.path.exists(f"results/xgboost/step2threshold_results_{exp}_w{window_size}.json"):
            utils.save_json({}, f"results/xgboost/step2threshold_results_{exp}_w{window_size}.json")
        results2 = utils.load_json(f"results/xgboost/step2threshold_results_{exp}_w{window_size}.json")
        # import pdb; pdb.set_trace()
        record = {
            'train_balanced_acc': train_balanced_acc2,
            'test_balanced_acc': test_balanced_acc2,
            'X_train_histogram': [[float(yy) for yy in xx] for xx in X_train_histogram.tolist()],
            'y_train_histogram': [int(xx) for xx in y_train_histogram],
            'X_test_histogram': [[float(yy) for yy in xx] for xx in X_test_histogram.tolist()],
            'y_test_histogram': [int(xx) for xx in y_test_histogram]
        }
        # print(record)
        results2[str(split_number)] = record
        utils.save_json(results2, f"results/xgboost/step2threshold_results_{exp}_w{window_size}.json")
        utils.save_json({'t1': t1, 't2': t2},
                        f'results/xgboost/{exp}/w{window_size}/step2threshold_{split_number}.json')

    if plot:
        plot_results(y2_pred_train, y2_train, y2_pred_test, y2_test,
                     f"results/img/{exp}_step2threshold_w{window_size}_s{split_number}.png")


# def step2_rank_corr(split_number, train_results, test_results, exp, plot=True, verbose=True, save=True,
#                     window_size=9):
#     y_test, y_pred_test, ids_test = test_results
#     # compute mean
#     mean_test, y2_test, ids_test = compute_mean(y_test, y_pred_test, ids_test)
#     # BELA BARTOK
#     # pair wise (framewise mean step 1 output, bartok number piece)
#     pair_wise = [(m, id2) for m, id2 in zip(mean_test, ids_test)]
#     # sort pieces by piece number
#     x = [y[0] for y in pair_wise]
#     y = [int(y[1]) for y in pair_wise]
#     # compute rho
#     sp_bartok = spearmanr(a=x, b=y)
#     kb_bartok = kendalltau(x, y, variant='b')
#     kc_bartok = kendalltau(x, y, variant='c')
#     # HENLE
#     # load henle dict
#     piece2henle = utils.load_json("mikrokosmos/henle_mikrokosmos.json")
#     # create triplet wise (mean ids and henle)
#     pair_wise = [(m, id2, piece2henle[str(id2)])for m, id2 in pair_wise]
#     # order by mean (1st index) and piece number (2nd index)
#     y = [int(y[2]) for y in pair_wise]
#     # compute rho
#     sp_henle = spearmanr(a=x, b=y)
#     kb_henle = kendalltau(x, y, variant='b')
#     kc_henle = kendalltau(x, y, variant='c')
#     if save:
#         if not os.path.exists(f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json"):
#             utils.save_json({}, f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json")
#         results2 = utils.load_json(f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json")
#         results2[str(split_number)] = {
#             'sp_bartok': sp_bartok,
#             'kb_bartok': kb_bartok,
#             'kc_bartok': kc_bartok,
#             'sp_henle': sp_henle,
#             'kb_henle': kb_henle,
#             'kc_henle': kc_henle,
#         }
#         print(results2[str(split_number)])
#         utils.save_json(results2, f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json")


def step2_rank_corr(split_number, train_results, test_results, exp, plot=True, verbose=True, save=True,
                    window_size=9):
    y_test, y_pred_test, ids_test = test_results
    # compute mean
    mean_test, y2_test, ids_test = compute_mean(y_test, y_pred_test, ids_test)
    # BELA BARTOK
    # pair wise (framewise mean step 1 output, bartok number piece)
    pair_wise = [(m, id2) for m, id2 in zip(mean_test, ids_test)]
    # sort pieces by piece number
    x = [y[0] for y in pair_wise]
    y = [int(y[1]) for y in pair_wise]
    # compute rho
    sp_bartok = spearmanr(a=x, b=y)
    kb_bartok = kendalltau(x, y, variant='b')
    kc_bartok = kendalltau(x, y, variant='c')
    # HENLE
    # load henle dict
    piece2henle = utils.load_json("mikrokosmos/henle_mikrokosmos.json")
    # create triplet wise (mean ids and henle)
    pair_wise = [(m, id2, piece2henle[str(id2)]) for m, id2 in pair_wise]
    # order by mean (1st index) and piece number (2nd index)
    y = [int(y[2]) for y in pair_wise]
    # compute rho
    sp_henle = spearmanr(a=x, b=y)
    kb_henle = kendalltau(x, y, variant='b')
    kc_henle = kendalltau(x, y, variant='c')
    if save:
        if not os.path.exists(f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json"):
            utils.save_json({}, f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json")
        results2 = utils.load_json(f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json")
        results2[str(split_number)] = {
            'sp_bartok': sp_bartok,
            'kb_bartok': kb_bartok,
            'kc_bartok': kc_bartok,
            'sp_henle': sp_henle,
            'kb_henle': kb_henle,
            'kc_henle': kc_henle,
        }
        print(results2[str(split_number)])
        utils.save_json(results2, f"results/xgboost/step2rank_corr_results_{exp}_w{window_size}.json")


# def step2_person(split_number, train_results, test_results, exp, plot=True, verbose=True, save=True,
#                     window_size=9):
#     y_test, y_pred_test, ids_test = test_results
#     # compute mean
#     mean_test, y2_test, ids_test = compute_mean(y_test, y_pred_test, ids_test)
#     # BELA BARTOK
#     # pair wise (framewise mean step 1 output, bartok number piece)
#     pair_wise = [(m, id2) for m, id2 in zip(mean_test, ids_test)]
#     # sort pieces by mean
#     pair_wise.sort(key=lambda y: y[0])
#     x = [y[1] for y in pair_wise]
#     # sort pieces by piece number
#     pair_wise.sort(key=lambda y: y[1])
#     y = [y[1] for y in pair_wise]
#     # compute rho
#     pr = pearsonr(x, y)
#     # HENLE
#     # load henle dict
#     piece2henle = utils.load_json("mikrokosmos/henle_mikrokosmos.json")
#     # create triplet wise (mean ids and henle)
#     pair_wise = [(m, id2, piece2henle[str(id2)])for m, id2 in pair_wise]
#     # order by mean (1st index) and piece number (2nd index)
#     pair_wise.sort(key=lambda y: (y[2], y[1]))
#     y = [y[1] for y in pair_wise]
#     # compute rho
#     pr_henle = pearsonr(x, y)
#     if save:
#         if not os.path.exists(f"results/xgboost/step2pearson_results_{exp}_w{window_size}.json"):
#             utils.save_json({}, f"results/xgboost/step2pearson_results_{exp}_w{window_size}.json")
#         results2 = utils.load_json(f"results/xgboost/step2pearson_results_{exp}_w{window_size}.json")
#         results2[str(split_number)] = {
#             'pearson_bartok': pr,
#             'pearson_henle': pr_henle
#         }
#         print(results2[str(split_number)])
#         utils.save_json(results2, f"results/xgboost/step2pearson_results_{exp}_w{window_size}.json")


def windowed(rang, experiment, ws, gpu, only_step2):
    X, y = load_rep(experiment)
    paths = load_rep_info(experiment)

    for split_number in range(rang):
        split = get_split(split_number, X, paths)
        train_results, test_results = step1(split,
                                            split_number=split_number,
                                            exp=experiment,
                                            window_size=ws,
                                            gpu=gpu,
                                            precomputed=only_step2)
        step2_rank_corr(split_number,
                        train_results,
                        test_results,
                        exp=experiment,
                        window_size=ws)

        step2_threshold(split_number,
                        train_results,
                        test_results,
                        exp=experiment,
                        window_size=ws)


def mkdir():
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/xgboost"):
        os.mkdir("results/xgboost")
    if not os.path.exists("results/img"):
        os.mkdir("results/img")
    for rep in ["rep_velocity", "rep_note", "rep_finger", "rep_finger_note", "rep_prob", "rep_finger_nakamura"]:
        for i in [1, 3, 5, 9, 13, 19, 27]:
            if not os.path.exists(f"results/xgboost/{rep}"):
                os.mkdir(f"results/xgboost/{rep}")
            if not os.path.exists(f"results/xgboost/{rep}/w{i}"):
                os.mkdir(f"results/xgboost/{rep}/w{i}")


def input_args(argv):
    if len(argv) != 5:
        raise "USE: python3 xgboost [representation] [windows_size] [device]"
    n_experiment = int(argv[1])
    if n_experiment == 1:
        experiment = "rep_velocity"
    elif n_experiment == 2:
        experiment = "rep_note"
    elif n_experiment == 3:
        experiment = "rep_finger"
    elif n_experiment == 4:
        experiment = "rep_finger_nakamura"
    elif n_experiment == 5:
        experiment = "rep_prob"
    elif n_experiment == 6:
        experiment = "rep_d_nakamura"
    elif n_experiment == 8:
        experiment = "rep_finger_note"
    else:
        raise "bad representation"

    window_size = int(argv[2])
    if not (1 <= window_size <= 100):
        raise "bad windows size"

    if argv[3].lower() == "cpu":
        gpu = False
    elif argv[3].lower() == "gpu":
        gpu = True
    else:
        raise "bad device: gpu or cpu"

    if argv[4] == "y":
        only_step2 = True
    elif argv[4] == "n":
        only_step2 = False
    else:
        raise "bad only step2"
    return experiment, window_size, gpu, only_step2


if __name__ == '__main__':
    mkdir()
    experiment, window_size, gpu, only_step2 = input_args(sys.argv)
    windowed(50, experiment, window_size, gpu, only_step2)
