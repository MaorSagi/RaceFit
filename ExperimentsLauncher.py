from argparse import ArgumentParser
import json
from typing import Union
import pandas as pd
from DataManager import init_data, get_raw_data_path, get_data_path, create_boolean_matrix, create_input_data, \
    data_cleaning_and_preprocessing, data_imputation
from Model import append_results_from_files, data_split, train, evaluate, get_models_path, fit_clustering_algorithm, \
    is_second_model_off, get_score_models_path, is_second_model_on, get_expr_loop_args
from utils import *
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost.compat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

params: dict[str: Union[str, int, tuple[str, object]]]  # Model hyper-parameters
clustering_algorithms = {"K-Means": KMeans}
models = {'AdaBoost': {'constructor': lambda: AdaBoostClassifier(),
                       'train': AdaBoostClassifier.fit,
                       'predict': AdaBoostClassifier.predict,
                       'predict_proba': AdaBoostClassifier.predict_proba},
          'Logistic': {'constructor': LogisticRegression,
                       'train': LogisticRegression.fit,
                       'predict': LogisticRegression.predict,
                       'predict_proba': LogisticRegression.predict_proba},
          'GradientBoosting': {'constructor': GradientBoostingClassifier,
                               'train': GradientBoostingClassifier.fit,
                               'predict': GradientBoostingClassifier.predict,
                               'predict_proba': GradientBoostingClassifier.predict_proba},
          'DecisionTree': {'constructor': lambda: DecisionTreeClassifier(),
                           'train': DecisionTreeClassifier.fit,
                           'predict': DecisionTreeClassifier.predict,
                           'predict_proba': DecisionTreeClassifier.predict_proba},
          'RandomForest': {'constructor': RandomForestClassifier,
                           'train': RandomForestClassifier.fit,
                           'predict': RandomForestClassifier.predict,
                           'predict_proba': RandomForestClassifier.predict_proba},
          'KNN': {'constructor': KNeighborsClassifier,
                  'train': KNeighborsClassifier.fit,
                  'predict': KNeighborsClassifier.predict,
                  'predict_proba': KNeighborsClassifier.predict_proba},
          'SVC': {'constructor': lambda: SVC(probability=True),
                  'train': SVC.fit,
                  'predict': SVC.predict,
                  'predict_proba': SVC.predict_proba},
          'XGBoost': {'constructor': lambda: XGBClassifier(objective="binary:logistic", use_label_encoder=False),
                      'train': XGBClassifier.fit,
                      'predict': XGBClassifier.predict,
                      'predict_proba': XGBClassifier.predict_proba},
          'LGBM': {'constructor': LGBMClassifier,
                   'train': LGBMClassifier.fit,
                   'predict': LGBMClassifier.predict,
                   'predict_proba': LGBMClassifier.predict_proba},
          'CatBoost': {'constructor': CatBoostClassifier,
                       'train': CatBoostClassifier.fit,
                       'predict': CatBoostClassifier.predict,
                       'predict_proba': CatBoostClassifier.predict_proba},
          'GaussianNB': {'constructor': GaussianNB,
                         'train': GaussianNB.fit,
                         'predict': GaussianNB.predict,
                         'predict_proba': GaussianNB.predict_proba},
          }

imputers = {'SimpleImputer': SimpleImputer, 'IterativeImputer': IterativeImputer,
            'KNNImputer': KNNImputer, 'without': None}
standardization_scalers = {'StandardScaler': StandardScaler, 'MinMaxScaler': MinMaxScaler,
                           'MaxAbsScaler': MaxAbsScaler, 'RobustScaler': RobustScaler, 'without': None}
similarity_functions = {'Cosine': cosine_similarity, 'Euclidean': euclidean_distances}
encoders = {'LabelEncoder': LabelEncoder}


def evaluate_popularity_baseline(iteration: int, X_test: pd.DataFrame, *_) -> dict[str: Union[str, float]]:
    return evaluate_baseline("cyclist_popularity_ranking_in_team", iteration, X_test)


def evaluate_popularity_in_continent_baseline(iteration: int, X_test: pd.DataFrame, *_) -> dict[str: Union[str, float]]:
    return evaluate_baseline("cyclist_popularity_ranking_in_continent_in_team", iteration, X_test)


feature_to_eval_function_dict = {"cyclist_popularity_ranking_in_team": evaluate_popularity_baseline,
                                 "cyclist_popularity_ranking_in_continent_in_team":
                                     evaluate_popularity_in_continent_baseline,
                                 "cyclist_popularity_ranking_in_race_class_in_team": None,
                                 "cyclist_popularity_ranking_in_race_type_in_team": None}


@time_wrapper
def evaluate_baseline(baseline_name: str, iteration: int, X_test: pd.DataFrame) -> dict[str:Union[str, float]]:
    overwrite, prediction_matrix_path, races_cyclist_matrix = init_parameters_for_eval_baselines()
    races_columns, races_cyclists_pred_matrix, races_cyclists_soft_pred_matrix, \
    races_cyclists_test_matrix = create_prediction_matrices(X_test, races_cyclist_matrix)
    grouped_data_by_races = X_test.groupby(RACE_ID_FEATURE)
    i = 0
    total_scores = {}
    race_cyclists = {}
    for race_id, g in grouped_data_by_races:
        top_cyclists = get_top_cyclists(baseline_name, g, race_id, races_columns, races_cyclists_test_matrix)

        race_cyclists[race_id] = set(
            [str(e) for e in X_test[X_test[RACE_ID_FEATURE] == race_id][CYCLIST_ID_FEATURE].values])
        cyclists_in_team = race_cyclists[race_id]
        set_prediction_matrix(i, race_id, races_columns, races_cyclists_pred_matrix, top_cyclists)
        set_prediction_proba_matrix(baseline_name, cyclists_in_team, g, i, race_id, races_columns,
                                    races_cyclists_soft_pred_matrix)
        i += 1
        total_scores = evaluate_results(params, iteration, race_cyclists, races_columns, races_cyclists_pred_matrix,
                                        races_cyclists_soft_pred_matrix, races_cyclists_test_matrix,
                                        prediction_matrix_path, log_path=get_data_path(params))
    set_metrics_array_to_str(total_scores, baseline_name)

    return total_scores


def init_parameters_for_eval_baselines() -> tuple[Union[str, pd.DataFrame], ...]:
    from DataManager import races_matrix_path
    races_cyclist_matrix = import_data_from_csv(races_matrix_path)
    data_path = get_data_path(params)
    prediction_matrix_path = f"{data_path}/prediction_matrix.csv"
    overwrite = params['overwrite']
    return overwrite, prediction_matrix_path, races_cyclist_matrix


def set_prediction_proba_matrix(baseline_feature: str, cyclists_in_team: set[int, ...], race_group: pd.DataFrame,
                                idx: int,
                                race_id: int, races_columns: list[str, ...],
                                races_cyclists_soft_pred_matrix: pd.DataFrame) -> None:
    races_cyclists_soft_pred_matrix.at[idx, RACE_ID_FEATURE] = race_id
    races_cyclists_soft_pred_matrix.loc[idx, races_columns[1:]] = 0
    for c in cyclists_in_team:
        races_cyclists_soft_pred_matrix.at[idx, c] = race_group[race_group[CYCLIST_ID_FEATURE] == float(c)].iloc[0][
            baseline_feature]


def set_prediction_matrix(idx: int, race_id: int, races_columns: list[str, ...],
                          races_cyclists_pred_matrix: pd.DataFrame,
                          top_cyclists: list[int, ...]) -> None:
    races_cyclists_pred_matrix.at[idx, RACE_ID_FEATURE] = race_id
    races_cyclists_pred_matrix.loc[idx, races_columns[1:]] = 0
    races_cyclists_pred_matrix.loc[idx, top_cyclists] = 1


def get_top_cyclists(baseline_feature: str, race_group: pd.DataFrame, race_id: int, races_columns: list[str, ...],
                     races_cyclists_test_matrix: pd.DataFrame) -> list[int, ...]:
    curr_race_pred = races_cyclists_test_matrix[RACE_ID_FEATURE] == race_id
    cyclists_to_choose = races_cyclists_test_matrix.loc[curr_race_pred, races_columns[1:]].sum(
        1)
    cyclists_participated_in_race_predict = race_group.groupby(CYCLIST_ID_FEATURE, as_index=False).mean().fillna(0)
    top_cyclists_indices = cyclists_participated_in_race_predict[baseline_feature].nlargest(
        cyclists_to_choose.values[0]).index
    top_cyclists = [str(c) for c in
                    cyclists_participated_in_race_predict.loc[top_cyclists_indices, CYCLIST_ID_FEATURE].values]
    return top_cyclists


def create_prediction_matrices(X_test: pd.DataFrame, races_cyclist_matrix: pd.DataFrame) \
        -> tuple[Union[list[str, ...], pd.DataFrame], ...]:
    races_cyclists_cpy = races_cyclist_matrix.copy()
    races_cyclists_test_matrix = races_cyclists_cpy.loc[
        races_cyclists_cpy[RACE_ID_FEATURE].isin(X_test[RACE_ID_FEATURE])]
    races_cyclists_test_matrix = races_cyclists_test_matrix.reset_index(drop=True)
    missing_cyclists_in_test = list(
        filter(lambda c: str(c) not in races_cyclists_test_matrix.columns, X_test[CYCLIST_ID_FEATURE].unique()))
    missing_cyclists_in_test = [str(c) for c in missing_cyclists_in_test]
    races_cyclists_test_matrix[missing_cyclists_in_test] = 0
    races_columns = races_cyclists_test_matrix.columns
    races_cyclists_pred_matrix = races_cyclists_test_matrix.copy()
    races_cyclists_pred_matrix[races_columns[1:]] = None
    races_cyclists_soft_pred_matrix = races_cyclists_pred_matrix.copy()
    races_cyclists_pred_matrix = races_cyclists_pred_matrix.fillna(0)
    return races_columns, races_cyclists_pred_matrix, races_cyclists_soft_pred_matrix, races_cyclists_test_matrix


def is_eval_and_results_exists(eval_functions: list[Callable], results_path: str, prediction_matrix_path: str) -> bool:
    if eval_functions is not None:
        if is_file_exists_and_should_not_be_changed(params['overwrite'], results_path):
            return True
        if is_file_exists_and_should_be_changed(params['overwrite'], results_path):
            os.remove(results_path)
        if is_file_exists_and_should_be_changed(params['overwrite'], prediction_matrix_path):
            os.remove(prediction_matrix_path)
    return False


def experiment_loop(results_path: str, prediction_matrix_path: str,
                    log_path: str, action_logging_name: str, eval_functions: list[Callable] = None,
                    train_func: Callable = None) -> None:
    from DataManager import important_races_ids
    if eval_functions or train_func:
        only_important = (params['only_important'] is not None) and params['only_important']
        data_exec_path = get_data_path(params)
        X, Y = import_data_from_csv(f'{data_exec_path}/X_cols_data.csv'), import_data_from_csv(
            f'{data_exec_path}/Y_cols_data.csv')
        iteration = 0
        cv_num_of_splits = params['kfold']
        group_by_feature = RACE_ID_FEATURE if params['data_split'] is None else params['data_split']
        split_args = (X, group_by_feature, cv_num_of_splits) if cv_num_of_splits else (X, group_by_feature)

        if is_eval_and_results_exists(eval_functions, results_path, prediction_matrix_path):
            return
        for train_index, test_index in data_split(params['data_split'], *split_args):
            try:
                X_train, X_test = X[X.index.isin(train_index)], X[
                    X.index.isin(test_index)]
                Y_train, Y_test = Y[Y.index.isin(train_index)], Y[
                    Y.index.isin(test_index)]
                train_time = None
                # for now the option of train and eval separately does not let to write time results
                if train_func is not None:
                    train_time, _ = train_func(iteration, params,
                                               X_train,
                                               Y_train)
                if eval_functions is not None:
                    for eval_function in eval_functions:
                        # ONLY VALID FOR TEST WITH ONE RACE
                        race_id = X_test[RACE_ID_FEATURE].unique()[0]
                        if (not only_important) or (only_important and (race_id in important_races_ids)):
                            non_baseline_args = (Y_test, params)
                            eval_time, results = eval_function(iteration, X_test, *non_baseline_args)

                            write_results_to_file(results_path, iteration + 1, train_time, eval_time,
                                                  results)

                log(f"{action_logging_name} Cross Validation: Iteration: %s" % (iteration + 1), "CV", log_path=log_path)
            except:
                log(f"Eval loop error, params: {get_params_str(params)}", "ERROR",
                    log_path=data_exec_path)
            iteration += 1
        log(f"Finished training and evaluating with params: {get_params_str(params)}", "Scores",
            log_path=log_path)


def run_job() -> None:
    clustering, create_input, create_stages_cyclists_matrix, eval_baselines, eval_model, k_clusters, \
    overwrite, preprocessing, train_eval, train_model = extract_main_job_parameters()
    non_team_dependent_actions = [create_stages_cyclists_matrix, clustering]
    team_dependent_actions = [create_input, preprocessing, eval_baselines, train_eval, train_model, eval_model]
    raw_data_exec_path = ''
    if any(non_team_dependent_actions):
        log(f"Started", log_path=f"{EXEC_PATH}")
    if any(team_dependent_actions):
        raw_data_exec_path = get_raw_data_path(params)
        log(f"Started", log_path=raw_data_exec_path)
    try:
        handle_job_use_cases(raw_data_exec_path, clustering, create_input,
                             create_stages_cyclists_matrix, eval_baselines,
                             eval_model, k_clusters, overwrite, preprocessing, train_eval,
                             train_model)

        if any(non_team_dependent_actions):
            log(f"Finished", log_path=f"{EXEC_PATH}")
        if any(team_dependent_actions):
            log(f"Finished", log_path=raw_data_exec_path)
    except:
        log(f"Main job, params: {get_params_str(params)}", "ERROR",
            log_path=raw_data_exec_path)


def handle_job_use_cases(raw_data_exec_path: str, clustering: bool, create_input: bool,
                         create_stages_cyclists_matrix: bool, eval_baselines: bool,
                         eval_model: bool, k_clusters: int, overwrite: bool, preprocessing: bool, train_eval: bool,
                         train_model: bool) -> None:
    data_exec_path=''
    if create_stages_cyclists_matrix:
        create_boolean_matrix()
    X_raw_data_path = f'{raw_data_exec_path}/X_cols_raw_data.csv'
    if create_input and is_file_does_not_exist_or_should_be_changed(overwrite, X_raw_data_path):
        create_input_data(params)
    if preprocessing or eval_baselines:
        data_exec_path = get_data_path(params)
    X_data_path = f'{data_exec_path}/X_cols_data.csv'
    if preprocessing and is_file_does_not_exist_or_should_be_changed(overwrite, X_data_path):
        Y_raw_data_path = f'{raw_data_exec_path}/Y_cols_raw_data.csv'
        Y_data_path = f'{data_exec_path}/Y_cols_data.csv'
        X, Y = import_data_from_csv(X_raw_data_path), import_data_from_csv(Y_raw_data_path)
        X, Y = data_cleaning_and_preprocessing(X, Y, params)
        X = data_imputation(params, X)
        X.to_csv(X_data_path, index=False, header=True)
        Y.to_csv(Y_data_path, index=False, header=True)
    if eval_baselines:
        baseline_results_path = f'{data_exec_path}/{BASELINES_FILE_NAME}'
        prediction_matrix_path = f"{data_exec_path}/prediction_matrix.csv"
        evaluate_baselines_functions = [feature_to_eval_function_dict[b] for b in evaluate_baselines]
        experiment_loop(baseline_results_path, prediction_matrix_path, data_exec_path, "Baseline",
                        evaluate_baselines_functions)
    if clustering:
        fit_clustering_algorithm(clustering_algorithms[CLUSTERING_ALG_NAME], k_clusters)
    if train_eval or train_model or eval_model:
        expr_loop_args = get_expr_loop_args(eval_model, train_eval, train_model, params)
        experiment_loop(*expr_loop_args)


def extract_main_job_parameters() -> tuple[Union[str, int], ...]:
    global params
    create_stages_cyclists_matrix = params['create_matrix']
    create_input = params['create_input']
    preprocessing = params['preprocessing']
    eval_baselines = params['eval_baselines']
    train_eval = params['train_eval']
    train_model = params['train_model']
    eval_model = params['eval_model']
    overwrite = params['overwrite']
    clustering = params['clustering']
    k_clusters = params['k_clusters']
    return clustering, create_input, create_stages_cyclists_matrix, eval_baselines, \
           eval_model, k_clusters, overwrite, preprocessing, train_eval, train_model


def write_results_to_file(results_path: str, iteration: int,
                          preprocess_and_train_time: float, eval_time: float,
                          results: dict[str: Union[str, float]]) -> None:
    from DataManager import team_names_dict
    total_scores = dict()
    total_scores['preprocess_and_train_time'] = preprocess_and_train_time
    total_scores['iteration'] = iteration
    total_scores['eval_time'] = eval_time
    total_scores.update(results)
    params_cpy = params.copy()
    params_cpy['team_name'] = team_names_dict[params_cpy['team_id']]
    for k in RAW_DATA_COLS + DATA_COLS + MODELS_COLS + SCORE_MODELS_COLS:
        if type(params_cpy[k]) is tuple:
            total_scores[k] = params_cpy[k][0]
        else:
            total_scores[k] = params_cpy[k]
    append_row_to_csv(results_path, total_scores)


def extract_exec_params() -> None:
    parser = init_exec_parser()
    init_params_from_parser(parser)


def init_exec_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-r', '--row-threshold', type=str)
    parser.add_argument('-c', '--col-threshold', type=str)
    parser.add_argument('-iw', '--imputer-workouts', type=str)
    parser.add_argument('-i', '--imputer', type=str)
    parser.add_argument('-s', '--scaler', type=str)
    parser.add_argument('-sim', '--similarity', type=str)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-sm', '--score-model', type=str)
    parser.add_argument('-sms', '--score-model-split', type=float)
    parser.add_argument('-kc', '--k-clusters', type=int)
    parser.add_argument('-k', '--kfold', type=int)
    parser.add_argument('-j', '--job-id', type=int)
    parser.add_argument('-a', '--action', type=str)
    parser.add_argument('-t', '--time-window', type=int)
    parser.add_argument('-af', '--aggregation-function', type=str)
    parser.add_argument('-o', '--overwrite', type=int)
    parser.add_argument('-pw', '--popularity-weights', type=str)
    parser.add_argument('-ws', '--workouts-source', type=str)
    parser.add_argument('-ds', '--data-split', type=str)
    parser.add_argument('-fi', '--feature-isolation', type=str)
    parser.add_argument('-ti', '--team-id', type=int)
    parser.add_argument('-oi', '--only-important', type=int)
    parser.add_argument('-rp', '--race-prediction', type=int)
    parser.add_argument('-rc', '--result-consideration', type=str)
    return parser


def init_params_from_parser(parser: ArgumentParser) -> None:
    global params
    from DataManager import aggregation_functions
    args = parser.parse_args()
    params = dict(row_threshold=float(args.row_threshold) if (
            args.row_threshold and check_float(args.row_threshold)) else args.row_threshold,
                  col_threshold=float(args.col_threshold) if (
                          args.col_threshold and check_float(args.col_threshold)) else args.col_threshold,
                  imputer_workouts=(
                      args.imputer_workouts, imputers[args.imputer_workouts]) if args.imputer_workouts else None,
                  imputer=(args.imputer, imputers[args.imputer]) if args.imputer else None,
                  scaler=(args.scaler, standardization_scalers[args.scaler]) if args.scaler else None,
                  time_window=args.time_window if args.time_window else None,
                  aggregation_function=(args.aggregation_function, aggregation_functions[
                      args.aggregation_function]) if args.aggregation_function else None,
                  similarity=(args.similarity, similarity_functions[args.similarity]) if args.similarity else None,
                  model=(args.model, models[args.model]) if args.model else None,
                  score_model=(args.score_model, models[args.score_model]) if (
                          args.score_model and args.score_model != 'without') else 'without',
                  kfold=args.kfold,
                  job_id=args.job_id,
                  create_matrix='create_matrix' in str(args.action),
                  preprocessing='preprocessing' in str(args.action),
                  create_input='create_input' in str(args.action),
                  import_input='import_input' in str(args.action),
                  clustering='clustering' in str(args.action),
                  train_eval='train_eval' in str(args.action),
                  train_model='train_model' in str(args.action),
                  eval_model='eval_model' in str(args.action),
                  popularity='popularity' in str(args.action),
                  append_results='append_results' in str(args.action),
                  eval_baselines='eval_baselines' in str(args.action),
                  overwrite=args.overwrite if args.overwrite else None,
                  popularity_weights=json.loads(args.popularity_weights) if args.popularity_weights else None,
                  workouts_source=args.workouts_source if args.workouts_source else STRAVA_SRC,
                  data_split=args.data_split,
                  feature_isolation=args.feature_isolation,
                  team_id=args.team_id,
                  only_important=args.only_important if args.only_important else None,
                  race_prediction=args.race_prediction if args.race_prediction else None,
                  result_consideration=args.result_consideration if args.result_consideration else None,
                  score_model_split=args.score_model_split if args.score_model_split else None,
                  k_clusters=args.k_clusters if args.k_clusters else None,
                  )


if __name__ == '__main__':
    create_dir_if_not_exist(EXECS_DIR_PATH)
    create_dir_if_not_exist(EXEC_PATH)
    create_dir_if_not_exist(ALLOCATION_MATRICES_PATH)
    if EXPR_TASK == APPEND_RESULTS:
        append_results_from_files(EXEC_PATH)
    else:
        extract_exec_params()
        race_prediction = params['race_prediction']
        workouts_source = params['workouts_source']
        team_id = params['team_id']
        init_data(team_id, workouts_source, race_prediction)
        run_job()
