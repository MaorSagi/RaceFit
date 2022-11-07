from argparse import ArgumentParser
import json
from typing import Union
from DataManager import init_data, get_raw_data_path, get_data_path, create_boolean_matrix, create_input_data, \
    data_cleaning_and_preprocessing, data_imputation
from Model import append_results_from_files, data_split, train, evaluate, get_models_path, fit_clustering_algorithm
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


def evaluate_popularity_baseline(iteration, X_test, *_):
    return evaluate_baseline("cyclist_popularity_ranking_in_team", iteration, X_test)


def evaluate_popularity_in_continent_baseline(iteration, X_test, *_):
    return evaluate_baseline("cyclist_popularity_ranking_in_continent_in_team", iteration, X_test)


feature_to_eval_function_dict = {"cyclist_popularity_ranking_in_team": evaluate_popularity_baseline,
                                 "cyclist_popularity_ranking_in_continent_in_team":
                                     evaluate_popularity_in_continent_baseline,
                                 "cyclist_popularity_ranking_in_race_class_in_team": None,
                                 "cyclist_popularity_ranking_in_race_type_in_team": None}


@time_wrapper
def evaluate_baseline(baseline_feature, iteration, X_test):
    from DataManager import races_matrix_path
    races_cyclist_matrix = import_data_from_csv(races_matrix_path)
    data_path = get_data_path(params)
    prediction_matrix_path = f"{data_path}/prediction_matrix.csv"
    if os.path.exists(prediction_matrix_path) and params['overwrite']:
        os.remove(prediction_matrix_path)
    races_cyclists_cpy = races_cyclist_matrix.copy()
    races_cyclists_test_matrix = races_cyclists_cpy.loc[races_cyclists_cpy['race_id'].isin(X_test['race_id'])]
    races_cyclists_test_matrix = races_cyclists_test_matrix.reset_index(drop=True)
    missing_cyclists_in_test = list(
        filter(lambda c: str(c) not in races_cyclists_test_matrix.columns, X_test['cyclist_id'].unique()))
    missing_cyclists_in_test = [str(c) for c in missing_cyclists_in_test]
    races_cyclists_test_matrix[missing_cyclists_in_test] = 0
    races_columns = races_cyclists_test_matrix.columns
    races_cyclists_pred_matrix = races_cyclists_test_matrix.copy()
    races_cyclists_pred_matrix[races_columns[1:]] = None
    races_cyclists_soft_pred_matrix = races_cyclists_pred_matrix.copy()

    grouped_data_by_races = X_test.groupby('race_id')
    i = 0
    races_cyclists_pred_matrix = races_cyclists_pred_matrix.fillna(0)
    total_scores = []
    race_cyclists = {}
    for race_id, g in grouped_data_by_races:
        curr_race_pred = races_cyclists_test_matrix['race_id'] == race_id
        cyclists_to_choose = races_cyclists_test_matrix.loc[curr_race_pred, races_columns[1:]].sum(
            1)
        cyclists_participated_in_race_predict = g.groupby('cyclist_id', as_index=False).mean().fillna(0)

        top_cyclists_indices = cyclists_participated_in_race_predict[baseline_feature].nlargest(
            cyclists_to_choose.values[0]).index
        top_cyclists = [str(c) for c in
                        cyclists_participated_in_race_predict.loc[top_cyclists_indices, 'cyclist_id'].values]

        race_cyclists[race_id] = set([str(e) for e in X_test[X_test['race_id'] == race_id]['cyclist_id'].values])
        cyclists_in_team = race_cyclists[race_id]

        # pred
        races_cyclists_pred_matrix.at[i, 'race_id'] = race_id
        races_cyclists_pred_matrix.loc[i, races_columns[1:]] = 0
        races_cyclists_pred_matrix.loc[i, top_cyclists] = 1
        # pred soft
        races_cyclists_soft_pred_matrix.at[i, 'race_id'] = race_id
        races_cyclists_soft_pred_matrix.loc[i, races_columns[1:]] = 0
        for c in cyclists_in_team:
            races_cyclists_soft_pred_matrix.at[i, c] = g[g['cyclist_id'] == float(c)].iloc[0][
                baseline_feature]

        i += 1

        total_scores = evaluate_results(params, iteration, race_cyclists, races_columns, races_cyclists_pred_matrix,
                                        races_cyclists_soft_pred_matrix, races_cyclists_test_matrix,
                                        prediction_matrix_path, log_path=get_data_path(params))

    total_scores['precision_recall_curve'] = json.dumps(list(total_scores['precision_recall_curve']))
    total_scores['roc_curve'] = json.dumps(list(total_scores['roc_curve']))
    total_scores['precisions'] = json.dumps(list(total_scores['precisions']))
    total_scores['recalls'] = json.dumps(list(total_scores['recalls']))
    total_scores['recalls_kn'] = json.dumps(list(total_scores['recalls_kn']))
    total_scores['baseline'] = baseline_feature

    # total_scores['norm_precisions'] = json.dumps(list(total_scores['norm_precisions']))
    # total_scores['norm_recalls'] = json.dumps(list(total_scores['norm_recalls']))

    return total_scores


def experiment_loop(results_path,
                    log_path, action_logging_name, eval_functions=None, train_func=None):
    from DataManager import important_races_ids
    if eval_functions or train_func:
        only_important = (params['only_important'] is not None) and params['only_important']
        data_exec_path = get_data_path(params)
        X, Y = import_data_from_csv(f'{data_exec_path}/X_cols_data.csv'), import_data_from_csv(
            f'{data_exec_path}/Y_cols_data.csv')
        iteration = 0
        cv_num_of_splits = params['kfold']
        group_by_feature = 'race_id' if params['data_split'] is None else params['data_split']
        split_args = (X, group_by_feature, cv_num_of_splits) if cv_num_of_splits else (X, group_by_feature)
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
                        race_id = X_test['race_id'].unique()[0]
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
    data_exec_path, raw_data_exec_path = '', ''
    if any(non_team_dependent_actions):
        log(f"Started", log_path=f"{EXEC_PATH}")
    if any(team_dependent_actions):
        data_exec_path = get_data_path(params)
        raw_data_exec_path = get_raw_data_path(params)
        log(f"Started", log_path=raw_data_exec_path)
    try:
        handle_job_use_cases(data_exec_path, raw_data_exec_path, clustering, create_input,
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


def handle_job_use_cases(data_exec_path: str, raw_data_exec_path: str, clustering: bool, create_input: bool,
                         create_stages_cyclists_matrix: bool, eval_baselines: bool,
                         eval_model: bool, k_clusters: int, overwrite: bool, preprocessing: bool, train_eval: bool,
                         train_model: bool) -> None:
    if create_stages_cyclists_matrix:
        create_boolean_matrix()
    X_raw_data_path = f'{raw_data_exec_path}/X_cols_raw_data.csv'
    if create_input and is_file_does_not_exist_or_should_be_changed(X_raw_data_path, overwrite):
        create_input_data(params)
    X_data_path = f'{data_exec_path}/X_cols_data.csv'
    if preprocessing and is_file_does_not_exist_or_should_be_changed(X_data_path, overwrite):
        Y_raw_data_path = f'{raw_data_exec_path}/Y_cols_raw_data.csv'
        Y_data_path = f'{data_exec_path}/Y_cols_data.csv'
        X, Y = import_data_from_csv(X_raw_data_path), import_data_from_csv(Y_raw_data_path)
        X, Y = data_cleaning_and_preprocessing(X, Y, params)
        X = data_imputation(params, X)
        X.to_csv(X_data_path, index=False, header=True)
        Y.to_csv(Y_data_path, index=False, header=True)
    if eval_baselines:
        baseline_results_path = f'{data_exec_path}/{BASELINES_FILE_NAME}'
        evaluate_baselines_functions = [feature_to_eval_function_dict[b] for b in evaluate_baselines]
        expr_loop_args = (baseline_results_path, data_exec_path, "Baseline", evaluate_baselines_functions)
        activate_experiment_loop(*expr_loop_args, baseline_results_path, overwrite)
    if clustering:
        fit_clustering_algorithm("K-Means", clustering_algorithms["K-Means"], k_clusters)
    if train_eval or train_model or eval_model:
        trained_model_path = get_models_path(params)
        model_results_path = f'{trained_model_path}/{MODEL_RESULTS_FILE_NAME}'
        evaluate_funcs = [evaluate] if (train_eval or eval_model) else None
        train_func = train if (train_eval or train_model) else None
        expr_loop_args = (model_results_path, trained_model_path, "Model", evaluate_funcs, train_func)
        activate_experiment_loop(*expr_loop_args, model_results_path, overwrite)


def activate_experiment_loop(expr_loop_args, model_results_path, overwrite):
    if is_file_does_not_exist_or_should_be_changed(model_results_path, overwrite):
        if is_file_exists_and_should_not_be_changed(model_results_path, overwrite):
            os.remove(model_results_path)
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


def write_results_to_file(results_path, iteration, preprocess_and_train_time, eval_time, results):
    from DataManager import team_names_dict
    total_scores = dict()
    total_scores['preprocess_and_train_time'] = preprocess_and_train_time
    total_scores['iteration'] = iteration
    total_scores['eval_time'] = eval_time
    total_scores.update(results)
    params_cpy = params.copy()
    params_cpy['team_name'] = team_names_dict[params_cpy['team_id']]
    for k in RAW_DATA_COLS + DATA_COLS + MODELS_COLS:
        if type(params_cpy[k]) is tuple:
            total_scores[k] = params_cpy[k][0]
        else:
            total_scores[k] = params_cpy[k]
    append_row_to_csv(results_path, total_scores)


def extract_exec_params():
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
    parser.add_argument('-sms', '--score-model-split', type=int)
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
                  score_model=args.score_model if args.score_model else None,
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
        init_data(team_id, workouts_source,race_prediction)
        run_job()
