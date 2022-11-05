import os
import time
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc, \
    roc_auc_score, precision_recall_curve
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import traceback
import re
from expr_consts import *
from datetime import datetime


def get_params_str(params):
    res = []
    for key in params.keys():
        if type(params[key]) is tuple:
            res += [params[key][0]]
        else:
            res += [params[key]]
    return str(res)


def create_dir_if_not_exist(save_file_path):
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)


def check_float(sting):
    return re.match(r'^-?\d+(?:\.\d+)$', sting) is not None


def check_int(sting):
    return re.match(r"[-+]?\d+(\.0*)?$", sting) is not None


def time_wrapper(func):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        delta_time = end_time - start_time
        return delta_time, result

    return wrap


def import_data_from_csv(name):
    file_path = name.replace('.csv', '') + '.csv'
    return pd.read_csv(file_path, low_memory=False) if os.path.exists(
        file_path) else pd.DataFrame()


def log(msg, type='INFO', debug=True, log_path=None):
    if LOG_DICT[type] <= LOG_DICT[LOG_LEVEL]:
        if log_path == None:
            log_path = EXEC_PATH
        with open(f'{log_path}/{type}_Logger.log', 'a+') as f:
            if type == 'ERROR':
                msg += f' ERROR DETAILS: {traceback.format_exc()}'
            f.write(str(datetime.now()) + '\t' + msg.replace('\n', '') + '\n')
            if debug:
                print(str(datetime.now()) + '\t' + msg)


def append_row_to_csv(file_path, row, columns=None):
    if columns is None:
        columns = list(row.keys())
    df = pd.DataFrame([row], columns=columns)
    file_exists = os.path.exists(file_path)
    if not file_exists:
        df.to_csv(file_path, header=True, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


def append_row_to_prediction_matrix(params, type, iter, row, races_columns, prediction_matrix_path):
    prediction_matrix_exists = os.path.exists(prediction_matrix_path)
    pred_df = None
    if prediction_matrix_exists:
        pred_df = pd.read_csv(prediction_matrix_path)
    write_to_file = ((not prediction_matrix_exists) or (
        pred_df[(pred_df['iteration'] == iter) & (pred_df['type'] == type)].empty))
    write_to_file = write_to_file or params['overwrite']
    if write_to_file:
        row_cpy = row.copy()
        missing = set(races_columns[1:]) - set(row_cpy.index)
        row_cpy = row_cpy.append(pd.Series({k: None for k in missing}))
        row_cpy['iteration'] = iter
        row_cpy['type'] = type
        row_cpy = row_cpy.sort_index(ascending=False)
        append_row_to_csv(prediction_matrix_path, row_cpy)


def update_scores(total_scores, ground_truth, row_pred):
    c_report = classification_report(ground_truth, row_pred, output_dict=True)

    total_scores['precision_1'] += c_report['1.0']['precision']
    total_scores['recall_1'] += c_report['1.0']['recall']
    total_scores['f1_score_1'] += c_report['1.0']['f1-score']
    total_scores['support_1'] += c_report['1.0']['support']

    total_scores['precision_0'] += c_report['0.0']['precision']
    total_scores['recall_0'] += c_report['0.0']['recall']
    total_scores['f1_score_0'] += c_report['0.0']['f1-score']
    total_scores['support_0'] += c_report['0.0']['support']

    total_scores['accuracy'] += c_report['accuracy']

    total_scores['macro_precision'] += c_report['macro avg']['precision']
    total_scores['macro_recall'] += c_report['macro avg']['recall']
    total_scores['macro_f1_score'] += c_report['macro avg']['f1-score']
    total_scores['macro_support'] += c_report['macro avg']['support']

    total_scores['weighted_precision'] += c_report['weighted avg']['precision']
    total_scores['weighted_recall'] += c_report['weighted avg']['recall']
    total_scores['weighted_f1_score'] += c_report['weighted avg']['f1-score']
    total_scores['weighted_support'] += c_report['weighted avg']['support']

    tn, fp, fn, tp = confusion_matrix(ground_truth, row_pred).ravel()
    total_scores['true_negative'] += tn
    total_scores['false_negative'] += fn
    total_scores['true_positive'] += tp
    total_scores['false_positive'] += fp
    total_scores['true_negative_rate'] += tn / (tn + fp) if (tn + fp) > 0 else 0
    total_scores['false_negative_rate'] += fn / (fn + tp) if (fn + tp) > 0 else 0
    total_scores['true_positive_rate'] += tp / (tp + fn) if (tp + fn) > 0 else 0
    total_scores['false_positive_rate'] += fp / (fp + tn) if (fp + tn) > 0 else 0

    return total_scores


def eval_special_metrics(total_scores, ground_truth, row_soft):
    precisions, recalls = [1], [0]
    recalls_kn = []
    # norm_precisions, norm_recalls = [1], [0]

    cyclists_pred_rankings_df = pd.DataFrame([{'cyclist_id': c, 'score': s} for c, s in row_soft.items()]).sort_values(
        'score', ascending=False).reset_index(drop=True)
    cyclists_participated = set(ground_truth[ground_truth == 1].index)
    # not_in_rankings
    # cyclists_participated =
    participated_cyclists_rankings = cyclists_pred_rankings_df[
        cyclists_pred_rankings_df['cyclist_id'].isin(cyclists_participated)].index
    MRR = sum(participated_cyclists_rankings) / len(participated_cyclists_rankings) if len(
        participated_cyclists_rankings) > 0 else NUM_OF_POINTS
    num_of_cyclists_in_race = len(cyclists_participated)
    for i in range(1, NUM_OF_POINTS):
        pred_i = cyclists_pred_rankings_df.iloc[:i]['cyclist_id'].values
        participated_and_pred_i = len(cyclists_participated.intersection(pred_i))
        precision_i = participated_and_pred_i / i
        # norm_base = min(i, num_of_cyclists_in_race) / i
        # norm_precision_i = precision_i / norm_base if norm_base > 0 else 0
        recall_i = participated_and_pred_i / num_of_cyclists_in_race if num_of_cyclists_in_race > 0 else 1
        # norm_recall_i = recall_i / norm_base if norm_base > 0 else 1
        precisions.append(precision_i)  # , norm_precisions.append(norm_precision_i)
        recalls.append(recall_i)  # , norm_recalls.append(norm_recall_i)

    for k in range(K_POINTS):
        pred_kn = cyclists_pred_rankings_df.iloc[:k + num_of_cyclists_in_race]['cyclist_id'].values
        participated_and_pred_kn = len(cyclists_participated.intersection(pred_kn))
        recall_kn = participated_and_pred_kn / num_of_cyclists_in_race if num_of_cyclists_in_race > 0 else 1
        recalls_kn.append(recall_kn)

    total_scores['cyclists_participated'] = len(cyclists_participated)
    total_scores['MRR'] += MRR
    # total_scores['norm_recalls'] += norm_recalls
    # total_scores['norm_precisions'] += norm_precisions
    total_scores['recalls'] += recalls
    total_scores['precisions'] += precisions
    total_scores['recalls_kn'] += recalls_kn

    return total_scores


def evaluate_results(params, iter, race_cyclists, races_columns, races_cyclists_pred_matrix,
                     races_cyclists_soft_pred_matrix, races_cyclists_test_matrix, prediction_matrix_path, log_path):
    total_scores = dict(MRR=0, recalls=np.zeros(NUM_OF_POINTS), precisions=np.zeros(NUM_OF_POINTS),
                        recalls_kn=np.zeros(K_POINTS),
                        # norm_recalls=np.zeros(NUM_OF_POINTS), norm_precisions=np.zeros(NUM_OF_POINTS),
                        precision_1=0, recall_1=0, f1_score_1=0, support_1=0, precision_0=0, recall_0=0, f1_score_0=0,
                        support_0=0, precision_recall_curve=np.zeros(NUM_OF_POINTS), roc_curve=np.zeros(NUM_OF_POINTS),
                        accuracy=0, macro_precision=0, macro_recall=0, macro_f1_score=0,
                        macro_support=0, weighted_precision=0, weighted_recall=0, weighted_f1_score=0,
                        weighted_support=0, true_negative=0, true_negative_rate=0, false_negative=0,
                        false_negative_rate=0, true_positive=0, true_positive_rate=0, false_positive=0,
                        false_positive_rate=0, prc_auc=0, roc_auc=0)
    for idx, row in races_cyclists_pred_matrix.iterrows():
        row_soft = races_cyclists_soft_pred_matrix.loc[idx]
        race_id = row['race_id']
        candidates = race_cyclists[race_id]

        row_soft = row_soft.loc[row_soft.index.intersection(candidates)].apply(float).sort_index()
        append_row_to_prediction_matrix(params, 'PROB', iter, row_soft, races_columns, prediction_matrix_path)

        row_pred = row.loc[row.index.intersection(candidates)].apply(float).sort_index()
        append_row_to_prediction_matrix(params, 'PRED', iter, row_pred, races_columns, prediction_matrix_path)

        ground_truth = races_cyclists_test_matrix.loc[idx][
            races_columns.intersection(candidates)].apply(float).sort_index()
        append_row_to_prediction_matrix(params, 'TRUE', iter, ground_truth, races_columns, prediction_matrix_path)

        recall_new_points, fpr_new_points = np.linspace(0, 1, NUM_OF_POINTS), np.linspace(0, 1, NUM_OF_POINTS)
        try:
            precision, recall, _ = precision_recall_curve(ground_truth, row_soft)
            interp = interp1d(recall, precision)
            precision_new_points = interp(recall_new_points)
            # wont work with more than 1 race in test
            total_scores['race_id'] = race_id
            total_scores['precision_recall_curve'] += precision_new_points

            fpr, tpr, _ = roc_curve(ground_truth, row_soft)
            interp = interp1d(fpr, tpr)
            tpr_new_points = interp(fpr_new_points)
            total_scores['roc_curve'] += tpr_new_points
            try:
                total_scores['roc_auc'] += roc_auc_score(ground_truth, row_soft)
            except:
                total_scores['roc_auc'] += 0
            total_scores['prc_auc'] += auc(recall_new_points, precision_new_points)

            # Cant mix continuous values in classification metrics
            # total_scores['precision'] += precision_score(ground_truth, row_soft)
            # total_scores['recall'] += recall_score(ground_truth, row_soft)

            # precision, recall, _ = precision_recall_curve(ground_truth, row_pred)
            # interp = interp1d(recall,precision)
            # precision_new_points = interp(recall_new_points)
            # total_scores['precision_recall_curve'] += precision_new_points
            total_scores = eval_special_metrics(total_scores, ground_truth, row_soft)
            total_scores = update_scores(total_scores, ground_truth, row_pred)


        except:
            log(f"Metrics scores error. Params: {get_params_str(params)}", "ERROR",
                log_path=log_path)
    total_scores = {k: v / len(races_cyclists_pred_matrix.index) for k, v in total_scores.items() if v is not None}
    return total_scores
