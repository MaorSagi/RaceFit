import json
import os
import time
from typing import Callable, Union, Literal

from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc, \
    roc_auc_score, precision_recall_curve
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import traceback
import re
from expr_consts import *
from datetime import datetime

def is_file_does_not_exist_or_should_be_changed(overwrite: bool, file_path: str) -> bool:
    return overwrite or (not os.path.exists(file_path))

def is_file_exists_and_should_not_be_changed(overwrite: bool, file_path: str) -> bool:
    return (not overwrite) and os.path.exists(file_path)

def is_file_exists_and_should_be_changed(overwrite: bool, file_path: str) -> bool:
    return overwrite and os.path.exists(file_path)


def get_params_str(params: dict[str: Union[str, int, tuple[str, object]]]) -> str:
    res = []
    for key in params.keys():
        if type(params[key]) is tuple:
            res += [params[key][0]]
        else:
            res += [params[key]]
    return str(res)


def create_dir_if_not_exist(save_file_path: str) -> None:
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)


def check_float(sting) -> bool:
    return re.match(r'^-?\d+(?:\.\d+)$', sting) is not None


def check_int(sting) -> bool:
    return re.match(r"[-+]?\d+(\.0*)?$", sting) is not None


def time_wrapper(func: Callable) -> Callable:
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        delta_time = end_time - start_time
        return delta_time, result

    return wrap


def import_data_from_csv(name: str) -> pd.DataFrame:
    file_path = name.replace('.csv', '') + '.csv'
    return pd.read_csv(file_path, low_memory=False) if os.path.exists(
        file_path) else pd.DataFrame()


def log(msg: str, log_type='INFO', debug=True, log_path: Union[None, str] = None) -> None:
    if LOG_DICT[log_type] <= LOG_DICT[LOG_LEVEL]:
        if log_path is None:
            log_path = EXEC_PATH
        with open(f'{log_path}/{log_type}_Logger.log', 'a+') as f:
            if log_type == 'ERROR':
                msg += f' ERROR DETAILS: {traceback.format_exc()}'
            f.write(str(datetime.now()) + '\t' + msg.replace('\n', '') + '\n')
            if debug:
                print(str(datetime.now()) + '\t' + msg)


def append_row_to_csv(file_path: str, row: dict[str, Union[str, int, float]], columns: list[str, ...] = None) -> None:
    if columns is None:
        columns = list(row.keys())
    df = pd.DataFrame([row], columns=columns)
    file_exists = os.path.exists(file_path)
    if not file_exists:
        df.to_csv(file_path, header=True, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


def append_row_to_prediction_matrix(params: dict[str: Union[str, int, tuple[str, object]]],
                                    record_type: Literal['PROB', 'PRED', 'TRUE'],
                                    iteration: int,
                                    row: pd.Series, races_columns: list[str, ...],
                                    prediction_matrix_path: str) -> None:
    prediction_matrix_exists = os.path.exists(prediction_matrix_path)
    pred_df = None
    if prediction_matrix_exists:
        pred_df = pd.read_csv(prediction_matrix_path)
    write_to_file = ((not prediction_matrix_exists) or (
        pred_df[(pred_df['iteration'] == iteration) & (pred_df['type'] == record_type)].empty))
    write_to_file = write_to_file or params['overwrite']
    if write_to_file:
        row_cpy = row.copy()
        missing = set(races_columns[1:]) - set(row_cpy.index)
        row_cpy = row_cpy.append(pd.Series({k: None for k in missing}))
        row_cpy['iteration'] = iteration
        row_cpy['type'] = record_type
        row_cpy = row_cpy.sort_index(ascending=False)
        append_row_to_csv(prediction_matrix_path, row_cpy)


def update_scores(total_scores: dict[str, Union[str, int, float]], ground_truth: pd.Series,
                  row_pred: pd.Series) -> \
        dict[str, Union[str, int, float]]:
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


def eval_special_metrics(total_scores: dict[str, Union[str, int, float]], ground_truth: pd.Series,
                         row_soft: pd.Series) -> dict[str, Union[str, int, float]]:
    cyclists_pred_rankings_df = get_cyclists_scores(row_soft)
    cyclists_participated = get_index_of_cyclists_participated(ground_truth)
    participated_cyclists_rankings = get_participated_cyclists_rankings(cyclists_participated,
                                                                        cyclists_pred_rankings_df)
    mrr = get_mean_reciprocal_rank(participated_cyclists_rankings)
    num_of_cyclists_in_race = len(cyclists_participated)
    precisions, recalls = get_recalls_and_precisions_at_i(cyclists_participated, cyclists_pred_rankings_df,
                                                          num_of_cyclists_in_race)
    recalls_kn = get_recall_kn(cyclists_participated, cyclists_pred_rankings_df, num_of_cyclists_in_race)

    total_scores['cyclists_participated'] = len(cyclists_participated)
    total_scores['MRR'] += mrr
    total_scores['recalls'] += recalls
    total_scores['precisions'] += precisions
    total_scores['recalls_kn'] += recalls_kn

    return total_scores


def get_recall_kn(cyclists_participated: set[int], cyclists_pred_rankings_df: pd.DataFrame,
                  num_of_cyclists_in_race: int) -> list[float, ...]:
    recalls_kn = []
    for k in range(K_POINTS):
        pred_kn = cyclists_pred_rankings_df.iloc[:k + num_of_cyclists_in_race][CYCLIST_ID_FEATURE].values
        participated_and_pred_kn = len(cyclists_participated.intersection(pred_kn))
        recall_kn = participated_and_pred_kn / num_of_cyclists_in_race if num_of_cyclists_in_race > 0 else 1
        recalls_kn.append(recall_kn)
    return recalls_kn


def get_recalls_and_precisions_at_i(cyclists_participated: set[int], cyclists_pred_rankings_df: pd.DataFrame,
                                    num_of_cyclists_in_race: int) -> tuple[list[float, ...], list[float, ...]]:
    precisions, recalls = [1], [0]
    for i in range(1, NUM_OF_POINTS):
        pred_i = cyclists_pred_rankings_df.iloc[:i][CYCLIST_ID_FEATURE].values
        participated_and_pred_i = len(cyclists_participated.intersection(pred_i))
        precision_i = participated_and_pred_i / i
        recall_i = participated_and_pred_i / num_of_cyclists_in_race if num_of_cyclists_in_race > 0 else 1
        precisions.append(precision_i)
        recalls.append(recall_i)
    return precisions, recalls


def get_mean_reciprocal_rank(participated_cyclists_rankings: list[int, ...]) -> float:
    return sum(participated_cyclists_rankings) / len(participated_cyclists_rankings) if len(
        participated_cyclists_rankings) > 0 else NUM_OF_POINTS


def get_participated_cyclists_rankings(cyclists_participated: set[int, ...], cyclists_pred_rankings_df: pd.DataFrame) -> \
        list[int, ...]:
    return cyclists_pred_rankings_df[cyclists_pred_rankings_df[CYCLIST_ID_FEATURE].isin(cyclists_participated)].index


def get_index_of_cyclists_participated(ground_truth: pd.Series) -> set[int, ...]:
    return set(ground_truth[ground_truth == 1].index)


def get_cyclists_scores(row_soft: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{CYCLIST_ID_FEATURE: c, 'score': s} for c, s in row_soft.items()]).sort_values(
        'score', ascending=False).reset_index(drop=True)

def set_metrics_array_to_str(total_scores: dict[str:Union[str, float]],baseline: str=None) -> None:
    total_scores['precision_recall_curve'] = json.dumps(list(total_scores['precision_recall_curve']))
    total_scores['roc_curve'] = json.dumps(list(total_scores['roc_curve']))
    total_scores['precisions'] = json.dumps(list(total_scores['precisions']))
    total_scores['recalls'] = json.dumps(list(total_scores['recalls']))
    total_scores['recalls_kn'] = json.dumps(list(total_scores['recalls_kn']))
    total_scores['baseline'] = baseline

def evaluate_results(params: dict[str: Union[str, int, tuple[str, object]]], iteration: int,
                     race_cyclists: dict[int, list[int, ...]],
                     races_columns: set[str], races_cyclists_pred_matrix: pd.DataFrame,
                     races_cyclists_soft_pred_matrix: pd.DataFrame, races_cyclists_test_matrix: pd.DataFrame,
                     prediction_matrix_path: str, log_path: str) -> dict[str, Union[str, int, float]]:
    total_scores = init_scores_dict()
    for idx, row in races_cyclists_pred_matrix.iterrows():
        row_soft = races_cyclists_soft_pred_matrix.loc[idx]
        race_id = row[RACE_ID_FEATURE]
        candidates = race_cyclists[race_id]
        ground_truth, row_pred, row_soft = append_predictions_to_pred_file(candidates, idx, iteration, params,
                                                                           prediction_matrix_path, races_columns,
                                                                           races_cyclists_test_matrix, row, row_soft)

        recall_new_points, fpr_new_points = np.linspace(0, 1, NUM_OF_POINTS), np.linspace(0, 1, NUM_OF_POINTS)
        total_scores[RACE_ID_FEATURE] = race_id
        try:
            total_scores, precision_new_points = eval_pr_curve(ground_truth, recall_new_points, row_soft, total_scores)
            total_scores = eval_roc(fpr_new_points, ground_truth, row_soft, total_scores)
            total_scores = eval_auc(ground_truth, precision_new_points, recall_new_points, row_soft, total_scores)
            total_scores = eval_special_metrics(total_scores, ground_truth, row_soft)
            total_scores = update_scores(total_scores, ground_truth, row_pred)

        except:
            log(f"Metrics scores error. Params: {get_params_str(params)}", "ERROR",
                log_path=log_path)
    total_scores = average_all_scores_in_iteration(races_cyclists_pred_matrix, total_scores)
    return total_scores


def average_all_scores_in_iteration(races_cyclists_pred_matrix: pd.DataFrame,
                                    total_scores: dict[str, Union[str, int, float]]) \
        -> dict[str, Union[str, int, float]]:
    return {k: v / len(races_cyclists_pred_matrix.index) for k, v in total_scores.items() if v is not None}


def eval_pr_curve(ground_truth: pd.Series, recall_new_points: list[int, ...], row_soft: pd.Series,
                  total_scores: dict[str, Union[str, int, float]]) -> dict[str, Union[str, int, float]]:
    precision, recall, _ = precision_recall_curve(ground_truth, row_soft)
    interp = interp1d(recall, precision)
    precision_new_points = interp(recall_new_points)
    # wont work with more than 1 race in test
    total_scores['precision_recall_curve'] += precision_new_points
    return total_scores, precision_new_points


def eval_roc(fpr_new_points: list[float, ...], ground_truth: pd.Series, row_soft: pd.Series,
             total_scores: dict[str, Union[str, int, float]]) -> dict[str, Union[str, int, float]]:
    fpr, tpr, _ = roc_curve(ground_truth, row_soft)
    interp = interp1d(fpr, tpr)
    tpr_new_points = interp(fpr_new_points)
    total_scores['roc_curve'] += tpr_new_points
    return total_scores


def eval_auc(ground_truth: pd.Series, precision_new_points: list[float, ...], recall_new_points: list[float, ...],
             row_soft: pd.Series, total_scores: dict[str, Union[str, int, float]]) -> dict[str, Union[str, int, float]]:
    try:
        total_scores['roc_auc'] += roc_auc_score(ground_truth, row_soft)
    except:
        total_scores['roc_auc'] += 0
    total_scores['prc_auc'] += auc(recall_new_points, precision_new_points)
    return total_scores


def append_predictions_to_pred_file(candidates: set[int, ...], idx: int, iteration: int,
                                    params: dict[str: Union[str, int, tuple[str, object]]],
                                    prediction_matrix_path: str, races_columns: set[str, ...],
                                    races_cyclists_test_matrix: pd.DataFrame, row: pd.Series,
                                    row_soft: pd.Series) -> tuple[pd.Series, ...]:
    row_soft = row_soft.loc[row_soft.index.intersection(candidates)].apply(float).sort_index()
    row_pred = row.loc[row.index.intersection(candidates)].apply(float).sort_index()
    ground_truth = races_cyclists_test_matrix.loc[idx][
        races_columns.intersection(candidates)].apply(float).sort_index()
    prediction_dict = {'PROB': row_soft, 'PRED': row_pred, 'TRUE': ground_truth}
    for record_type, row in prediction_dict.items():
        append_row_to_prediction_matrix(params, record_type, iteration, row, races_columns, prediction_matrix_path)
    return ground_truth, row_pred, row_soft


def init_scores_dict() -> dict:
    return dict(MRR=0, recalls=np.zeros(NUM_OF_POINTS), precisions=np.zeros(NUM_OF_POINTS),
                recalls_kn=np.zeros(K_POINTS), precision_1=0, recall_1=0, f1_score_1=0, support_1=0,
                precision_0=0, recall_0=0, f1_score_0=0,
                support_0=0, precision_recall_curve=np.zeros(NUM_OF_POINTS), roc_curve=np.zeros(NUM_OF_POINTS),
                accuracy=0, macro_precision=0, macro_recall=0, macro_f1_score=0,
                macro_support=0, weighted_precision=0, weighted_recall=0, weighted_f1_score=0,
                weighted_support=0, true_negative=0, true_negative_rate=0, false_negative=0,
                false_negative_rate=0, true_positive=0, true_positive_rate=0, false_positive=0,
                false_positive_rate=0, prc_auc=0, roc_auc=0)
