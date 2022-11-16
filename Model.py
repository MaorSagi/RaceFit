import ast
import pickle
import joblib
import pandas as pd

from utils import *


def leave_one_out(df: pd.DataFrame, group_id: str) -> list[tuple[list[int], ...]]:
    train_indices, test_indices = [], []
    df_grouped = df.groupby(group_id, sort=False)
    for idx, group in df_grouped:
        train_idx, test_idx = (set(df.index) - set(group.index)), set(group.index)
        train_indices.append(train_idx)
        test_indices.append(test_idx)
    return zip(train_indices, test_indices)


def grouped_time_series_split(df: pd.DataFrame, group_id: int, cv_num_of_splits: int = None) -> list[
    tuple[list[int], ...]]:
    number_of_groups = len(df[group_id].unique())
    if cv_num_of_splits is None:
        cv_num_of_splits = number_of_groups
    train_indices, test_indices = [], []
    # the num of stages in test does not changes in TimeSeriesSplit so I did the same..
    if cv_num_of_splits == 1:
        num_of_groups_in_test = int(number_of_groups / 5)
    else:
        num_of_groups_in_test = int(number_of_groups / cv_num_of_splits)
    for spilt_idx in range(cv_num_of_splits):
        group_idx = 0
        train_idx, test_idx = [], []
        for idx, group in df[::-1].groupby(group_id, sort=False):
            if group_idx >= spilt_idx * num_of_groups_in_test:
                # Assuming the first races are the most recent in Y df
                if group_idx - spilt_idx * num_of_groups_in_test < num_of_groups_in_test:
                    test_idx.extend(group.index)
                else:
                    train_idx.extend(group.index)
            group_idx += 1
        train_indices.append(train_idx)
        test_indices.append(test_idx)
        number_of_groups -= num_of_groups_in_test
        if cv_num_of_splits == 1:
            break
    return zip(train_indices, test_indices)


def data_split(split_type: Literal[YEAR_SPLIT, LEAVE_ONE_OUT, GROUP_SPLIT] = None, *args) -> list[
    tuple[list[int], ...]]:
    if split_type == YEAR_SPLIT:
        df, group_id, n = args
        df[group_id] = df[RACE_ID_FEATURE].map()
        res = grouped_time_series_split(df, group_id, n)
        return res
    if split_type == LEAVE_ONE_OUT:
        return leave_one_out(*args)
    return grouped_time_series_split(*args)


def miss_rate(df: pd.DataFrame) -> dict[str:float]:
    miss_rate_dict = {}
    for col in df.columns:
        miss_rate_dict[col] = round(100 * (len(df[df[col].isna()]) / len(df)))
    return sorted(miss_rate_dict.items(), key=lambda item: item[1])


def get_team_ranking_bins_for_stage(stage_id: int) -> pd.DataFrame:
    from DataManager import teams_rankings
    return teams_rankings[teams_rankings[STAGE_ID_FEATURE] == stage_id]


def duplicate_high_results_records(X: pd.DataFrame, y: pd.DataFrame, times_to_mul_list_str: str) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    X_addition = pd.DataFrame(columns=X.columns)
    y_addition = pd.Series()
    times_to_mul_list: list[int, ...] = eval(times_to_mul_list_str)
    n_bins = len(times_to_mul_list)
    for s_id, g in X.groupby(STAGE_ID_FEATURE):
        team_ranking_for_stage = get_team_ranking_bins_for_stage(s_id)
        if team_ranking_for_stage.empty:
            continue
        teams_rankings_row = team_ranking_for_stage.iloc[0]
        cutoffs_str = teams_rankings_row[f'cutoffs_{n_bins}']
        if str(cutoffs_str) == 'nan':
            continue
        cutoffs = eval(cutoffs_str)
        ranking = teams_rankings_row[f'ranking']
        times_to_mul = get_times_to_multiply_the_record(cutoffs, ranking, times_to_mul_list)
        X_addition, y_addition = mul_records(X_addition, g, times_to_mul, y, y_addition)
    return X.append(X_addition, ignore_index=True), y.append(y_addition, ignore_index=True)


def mul_records(X_addition: pd.DataFrame, df_to_dup: pd.DataFrame, times_to_mul: int, y: pd.DataFrame,
                y_addition: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    for t in range(times_to_mul - 1):
        X_addition = X_addition.append(df_to_dup, ignore_index=True)
        y_addition = y_addition.append(y[df_to_dup.index], ignore_index=True)
    return X_addition, y_addition


def get_times_to_multiply_the_record(cutoffs: list[int], ranking: int, times_to_mul_list: list[int]) -> int:
    times_to_mul = times_to_mul_list[-1]
    i = 0
    for c in cutoffs:
        if ranking <= c:
            times_to_mul = times_to_mul_list[i]
            break
        i += 1
    return times_to_mul


def transform_labels_from_stages_to_races(X_s, y_s, cyclist_stage_predict, params, stage_types_total):
    X, y = [], []
    for (race, cyclist), cyclist_in_race in X_s.groupby([RACE_ID_FEATURE, CYCLIST_ID_FEATURE]):
        participated_in_race = int(all(y_s.loc[cyclist_in_race.index]))
        y.append(participated_in_race)
        input_row = {}
        for stage_type in range(stage_types_total):
            input_row[RACE_ID_FEATURE]=race
            input_row[CYCLIST_ID_FEATURE]=cyclist
            stages_in_cluster = cyclist_in_race[cyclist_in_race['cluster'] == stage_type]
            input_row[f'days_in_stage_type_{stage_type}'] = len(stages_in_cluster)
            cyclist_stage_input = stages_in_cluster.copy()
            cyclist_stage_input = drop_unnecessary_features(cyclist_stage_input, params)
            if len(stages_in_cluster) > 0:
                participation_pred = np.mean([p[1] for p in cyclist_stage_predict(cyclist_stage_input)])
            else:
                participation_pred = 0
            input_row[f'mean_score_of_stage_type_{stage_type}'] = participation_pred
        X.append(input_row)
    X_t = pd.DataFrame(X)
    y_t = pd.Series(y)
    return X_t, y_t


def sort_transform_data(X, y, cyclist_stage_predict, params):
    X_s,y_s = X.copy(),y.copy()
    k_clusters = params['k_clusters']
    c_model = load_clusters(k_clusters)
    samples_clusters=c_model.predict(X_s[CLUSTERS_FEATURES])
    X_s['cluster'] = pd.Series(samples_clusters)
    X_s, y_s = transform_labels_from_stages_to_races(X_s, y_s, cyclist_stage_predict, params, k_clusters)
    return X_s, y_s


@time_wrapper
def train(iteration: int, params: dict[str: Union[str, int, tuple[str, object]]],
          cyclists: pd.DataFrame,
          stages: pd.DataFrame) -> None:
    trained_model_path = get_models_path(params)
    X, y, model_name, overwrite, result_consideration = extract_training_parameters(
        cyclists, params, stages)
    X, y, X_s, y_s = handle_score_model_split(params, X, y)

    fit_model(X, y, iteration, model_name, overwrite, params, result_consideration,
              trained_model_path,
              MODEL)
    train_second_level_model = X_s is not None
    if train_second_level_model:
        cyclist_stage_predict = lambda x: model_predict(x, iteration, model_name, params, trained_model_path, MODEL)
        X_s, y_s = sort_transform_data(X_s, y_s, cyclist_stage_predict, params)
        fit_model(X_s, y_s, iteration, model_name, overwrite, params, result_consideration,
                  trained_model_path,
                  SCORE_MODEL)


def fit_model(X, y, iteration, model_name, overwrite, params, result_consideration,
              trained_model_path,
              model_type: Literal[MODEL, SCORE_MODEL]) -> None:
    try:
        model_filename = f"{model_type}_{iteration + 1}_{model_name}.pkl"
        model_full_path = f'{trained_model_path}/{model_filename}'
        if is_file_exists_and_should_not_be_changed(overwrite, model_full_path):
            return
        model = params[model_type][1]['constructor']()
        if X.empty:
            return
        X_new = drop_unnecessary_features(X, params)
        if result_consideration:
            X_new, y = duplicate_high_results_records(X_new, y, result_consideration)
        train_func = params[model_type][1]['train']
        train_func(model, X_new, y)
        pickle.dump(model, open(model_full_path, 'wb'))
    except Exception as err:
        model_exec_path = get_models_path(params)
        log(f"Params: {get_params_str(params)},Train : {err}\n {traceback.format_exc()}", "ERROR",
            log_path=model_exec_path)


def drop_unnecessary_features(X: pd.DataFrame,
                              params: dict[str: Union[str, int, tuple[str, object]]]) -> pd.DataFrame:
    X_new = X.copy()
    features_to_remove = get_features_to_remove(X_new, params) + IDS_COLS + ['cluster']
    X_new = X_new.drop(columns=set(X_new.columns).intersection(features_to_remove))
    return X_new


def handle_score_model_split(params, X, y) -> tuple[Union[float, pd.DataFrame], ...]:
    X_s, y_s = None, None
    if params['score_model']:
        if params['score_model_split'] is None:
            raise ValueError("Invalid input: for score model option the user should specify the score model split.")
        split_fraction = params['score_model_split']
        races_ids = X[RACE_ID_FEATURE].drop_duplicates()
        sampled_races_ids = races_ids.sample(frac=split_fraction, random_state=1)
        X_s = X[X[RACE_ID_FEATURE].isin(sampled_races_ids.values)]
        y_s = y.loc[X_s.index]
        X = X[~X[RACE_ID_FEATURE].isin(sampled_races_ids.values)]
        y = y.loc[X.index]
    return X, y, X_s, y_s


def extract_training_parameters(cyclists: pd.DataFrame, params: dict[str: Union[str, int, tuple[str, object]]],
                                stages: pd.DataFrame) -> tuple[Union[pd.DataFrame, list[str, ...], str, bool], ...]:
    overwrite = params['overwrite']
    result_consideration = params['result_consideration']
    model_name = params['model'][0]
    y = stages['participated']
    X = pd.concat([cyclists, stages], axis=1).drop(columns='participated')
    return X, y, model_name, overwrite, result_consideration


def fill_pred_matrix(X_test, y_test, y_prob, pred_matrix, group_feature):
    from DataManager import get_race_date_by_race, get_teams_cyclists_in_year, get_race_date_by_stage
    for event_id, g in X_test.groupby(group_feature):
        get_date_func = get_race_date_by_race if group_feature == RACE_ID_FEATURE else get_race_date_by_stage
        race_date = get_date_func(event_id)
        cyclists = get_teams_cyclists_in_year(race_date.year)

        for i, r in g.iterrows():
            cyclist = r[CYCLIST_ID_FEATURE]
            if cyclist in cyclists[CYCLIST_ID_FEATURE].values:
                cyclist_in_team = cyclists[cyclist == cyclists[CYCLIST_ID_FEATURE]].iloc[0]
                if cyclist_in_team['start_date'] <= race_date <= cyclist_in_team['stop_date']:
                    cyclist_column = str(cyclist)
                    event_idx = \
                        pred_matrix[pred_matrix[group_feature] == event_id].index[
                            0]
                    pred_idx = y_test.index.get_loc(i)
                    pred_matrix.loc[event_idx, cyclist_column] = y_prob[pred_idx][1]
        return pred_matrix


@time_wrapper
def evaluate(iteration: int, X_test: pd.DataFrame, y_test: pd.DataFrame,
             params: dict[str: Union[str, int, tuple[str, object]]]) -> dict[str:Union[float, list[float], str], ...]:
    model_name, race_prediction, races_cyclist_matrix, \
    stages_cyclist_matrix, trained_model_path = init_parameters_for_eval_func(params)

    stages_cyclists_pred_matrix, _ = init_pred_matrix(X_test, stages_cyclist_matrix, STAGE_ID_FEATURE, race_prediction)
    races_cyclists_pred_matrix, races_cyclists_test_matrix = init_pred_matrix(X_test, races_cyclist_matrix,
                                                                              RACE_ID_FEATURE)
    races_cyclists_pred_matrix = races_cyclists_pred_matrix.fillna(0)
    races_cyclists_soft_pred_matrix = races_cyclists_pred_matrix.copy()
    cyclists_columns = races_cyclists_test_matrix.columns[1:]

    second_level_model_off = not params['score_model']
    if second_level_model_off:
        X_test_as_input = drop_unnecessary_features(X_test, params)
        y_prob = model_predict(X_test_as_input, iteration, model_name, params, trained_model_path, MODEL)
    else:
        cyclist_stage_predict = lambda x: model_predict(x, iteration, model_name, params, trained_model_path, MODEL)
        X_s, y_s = sort_transform_data(X_test, y_test, cyclist_stage_predict, params)
        y_prob = model_predict(X_s, iteration, model_name, params, trained_model_path, SCORE_MODEL)
        race_prediction = True
        X_test, y_test = X_s, y_s

    races_cyclists_soft_pred_matrix, stages_cyclists_pred_matrix = fill_pred_matrices(X_test, params,
                                                                                      race_prediction,
                                                                                      races_cyclists_soft_pred_matrix,
                                                                                      stages_cyclists_pred_matrix,
                                                                                      y_prob, y_test)
    total_scores = evaluate_model_predictions(X_test, cyclists_columns, iteration, params, race_prediction,
                                              races_cyclists_pred_matrix, races_cyclists_soft_pred_matrix,
                                              races_cyclists_test_matrix, stages_cyclists_pred_matrix,
                                              trained_model_path, y_prob)

    return total_scores


def evaluate_model_predictions(X_test: pd.DataFrame, cyclists_columns: list[int, ...], iteration: int,
                               params: dict[str: Union[str, int, tuple[str, object]]], race_prediction: bool,
                               races_cyclists_pred_matrix: pd.DataFrame, races_cyclists_soft_pred_matrix: pd.DataFrame,
                               races_cyclists_test_matrix: pd.DataFrame, stages_cyclists_pred_matrix: pd.DataFrame,
                               trained_model_path: str, y_prob: list[float, ...]):
    grouped_data_by_races = X_test.groupby(RACE_ID_FEATURE)
    race_cyclists = {}
    i = 0
    total_scores = []
    for race_id, g in grouped_data_by_races:
        cyclists_to_choose = get_number_of_cyclists_participated_in_race(cyclists_columns, race_id,
                                                                         races_cyclists_test_matrix)
        race_cyclists[race_id] = get_race_cyclists_in_input(X_test, race_id)
        cyclists_in_team = race_cyclists[race_id]

        cyclists_participated_in_race_predict = get_cyclists_prediction_in_race(cyclists_columns, cyclists_in_team, g,
                                                                                i, race_prediction,
                                                                                races_cyclists_soft_pred_matrix,
                                                                                stages_cyclists_pred_matrix)

        top_cyclists = get_top_cyclists(cyclists_participated_in_race_predict, cyclists_to_choose, y_prob)

        set_prediction_matrix(cyclists_columns, i, race_id, races_cyclists_pred_matrix, top_cyclists)
        set_prediction_proba_matrix(cyclists_columns, i, race_id, races_cyclists_soft_pred_matrix, y_prob, top_cyclists,
                                    cyclists_in_team, cyclists_participated_in_race_predict)

        i += 1
        prediction_matrix_path = f"{trained_model_path}/prediction_matrix.csv"
        total_scores = evaluate_results(params, iteration, race_cyclists, cyclists_columns, races_cyclists_pred_matrix,
                                        races_cyclists_soft_pred_matrix, races_cyclists_test_matrix,
                                        prediction_matrix_path, log_path=trained_model_path)
    set_metrics_array_to_str(total_scores)
    return total_scores


def set_prediction_proba_matrix(cyclists_columns: list[str, ...], race_idx: int, race_id: int,
                                races_cyclists_soft_pred_matrix: pd.DataFrame, y_prob: pd.Series,
                                top_cyclists: list[int, ...],
                                cyclists_in_team: set[int, ...],
                                cyclists_participated_in_race_predict: pd.DataFrame) -> None:
    races_cyclists_soft_pred_matrix.at[race_idx, RACE_ID_FEATURE] = race_id
    races_cyclists_soft_pred_matrix.loc[race_idx, cyclists_columns[1:]] = 0
    if y_prob is None:
        races_cyclists_soft_pred_matrix.loc[race_idx, top_cyclists] = 1
    for c in cyclists_in_team:
        races_cyclists_soft_pred_matrix.at[race_idx, c] = cyclists_participated_in_race_predict[c]


def set_prediction_matrix(cyclists_columns: list[str], race_idx: int, race_id: int,
                          races_cyclists_pred_matrix: pd.DataFrame, top_cyclists: list[str]) -> None:
    races_cyclists_pred_matrix.at[race_idx, RACE_ID_FEATURE] = race_id
    races_cyclists_pred_matrix.loc[race_idx, cyclists_columns[1:]] = 0
    races_cyclists_pred_matrix.loc[race_idx, top_cyclists] = 1


def get_race_cyclists_in_input(X_test: pd.DataFrame, race_id: int) -> set[str, ...]:
    curr_race_pred_in_X = X_test[RACE_ID_FEATURE] == race_id
    return set([str(e) for e in X_test[curr_race_pred_in_X][CYCLIST_ID_FEATURE].values])


def get_cyclists_prediction_in_race(cyclists_columns: list[str, ...], cyclists_in_team: set[int], race_group,
                                    race_idx: int, race_prediction: bool,
                                    races_cyclists_soft_pred_matrix: pd.DataFrame,
                                    stages_cyclists_pred_matrix: pd.DataFrame) -> pd.Series:
    if race_prediction:
        cyclists_participated_in_race_predict = races_cyclists_soft_pred_matrix.loc[
            race_idx, set.intersection(set(cyclists_columns), cyclists_in_team)].fillna(0)
    else:
        stages_in_race = race_group[STAGE_ID_FEATURE].unique()
        curr_race_stages_pred = stages_cyclists_pred_matrix[STAGE_ID_FEATURE].isin(stages_in_race)
        cyclists_participated_in_race_predict = stages_cyclists_pred_matrix.loc[
            curr_race_stages_pred, set.intersection(set(cyclists_columns), cyclists_in_team)]
        cyclists_participated_in_race_predict = cyclists_participated_in_race_predict.mean().fillna(0)
    return cyclists_participated_in_race_predict


def get_top_cyclists(cyclists_participated_in_race_predict: pd.Series, cyclists_to_choose: int, y_prob: pd.Series):
    if y_prob is None:
        top_cyclists = cyclists_participated_in_race_predict.sample(cyclists_to_choose).index
    else:
        top_cyclists = cyclists_participated_in_race_predict.nlargest(cyclists_to_choose).index
    return top_cyclists


def get_number_of_cyclists_participated_in_race(cyclists_columns: list[int, ...], race_id: int,
                                                races_cyclists_test_matrix: pd.DataFrame) -> int:
    curr_race_pred = races_cyclists_test_matrix[RACE_ID_FEATURE] == race_id
    cyclists_to_choose = races_cyclists_test_matrix.loc[curr_race_pred, cyclists_columns].sum(
        1).values[0]
    return cyclists_to_choose


def fill_pred_matrices(X_test, params, race_prediction, races_cyclists_soft_pred_matrix, stages_cyclists_pred_matrix,
                       y_prob, y_test):
    try:
        if y_prob is not None:
            if race_prediction:
                races_cyclists_soft_pred_matrix = fill_pred_matrix(X_test, y_test, y_prob,
                                                                   races_cyclists_soft_pred_matrix, RACE_ID_FEATURE)
            else:
                stages_cyclists_pred_matrix = fill_pred_matrix(X_test, y_test, y_prob,
                                                               stages_cyclists_pred_matrix,
                                                               STAGE_ID_FEATURE)
    except Exception as err:
        log(f"Evaluate function: Error: {err}, prediction job, params: {get_params_str(params)}", "ERROR",
            log_path=get_models_path(params))
    return races_cyclists_soft_pred_matrix, stages_cyclists_pred_matrix


def model_predict(X_test_as_input: pd.DataFrame, iteration: int, model_name: str,
                  params: dict[str: Union[str, int, tuple[str, object]]], trained_model_path: str,
                  model_type: Literal[MODEL, SCORE_MODEL]) -> np.array:
    y_prob = None
    try:
        model_filename = f"{model_type}_{iteration + 1}_{model_name}.pkl"
        model_full_path = f'{trained_model_path}/{model_filename}'
        if os.path.exists(model_full_path):
            model = pickle.load(open(model_full_path, 'rb'))
            y_prob = params[model_type][1]['predict_proba'](model, X_test_as_input)
    except Exception as err:
        log(f"Predict stage value. Error: {err} Params: {get_params_str(params)}", "ERROR",
            log_path=trained_model_path)
    return y_prob


def get_features_to_remove(X_test_as_input: pd.DataFrame, params: dict[str: Union[str, int, tuple[str, object]]]) -> \
        list[str]:
    from DataManager import cyclist_stats_cols
    feature_isolation = params['feature_isolation']
    if feature_isolation is not None:
        features_to_isolate = set(X_test_as_input.columns).intersection(
            cyclist_stats_cols + list(['race_total_distance',
                                       'race_total_elevation_gain']))
        if feature_isolation == 'without_new_features':
            features_to_remove = features_to_isolate
        else:
            features_to_remove = list(filter(lambda f: f != feature_isolation, features_to_isolate))
    else:
        features_to_remove = []
    # features_to_remove = ['distance_from_last_workout', 'distance_from_last_race', 'cyclist_weeks_since_last_race']
    return features_to_remove


def init_pred_matrix(X_test: pd.DataFrame, cyclist_allocation_matrix: pd.DataFrame,
                     event_id_feature: str, race_prediction: bool = False) -> tuple[Union[pd.DataFrame, None], ...]:
    if (not race_prediction) or event_id_feature == RACE_ID_FEATURE:
        cyclist_allocation_cpy = cyclist_allocation_matrix.copy()
        cyclist_allocation_test_matrix = cyclist_allocation_cpy.loc[
            cyclist_allocation_cpy[event_id_feature].isin(X_test[event_id_feature])]
        cyclist_allocation_test_matrix = cyclist_allocation_test_matrix.reset_index(drop=True)
        missing_cyclists_in_test = list(
            filter(lambda c: str(c) not in cyclist_allocation_test_matrix.columns, X_test[CYCLIST_ID_FEATURE].unique()))
        missing_cyclists_in_test = [str(c) for c in missing_cyclists_in_test]
        cyclist_allocation_test_matrix[missing_cyclists_in_test] = 0
        cyclist_allocation_pred_matrix = cyclist_allocation_test_matrix.copy()
        cyclist_allocation_pred_matrix[cyclist_allocation_pred_matrix.columns[1:]] = None
        return cyclist_allocation_pred_matrix, cyclist_allocation_test_matrix
    return None, None


def init_parameters_for_eval_func(params: dict[str: Union[str, int, tuple[str, object]]]) -> tuple[
    Union[str, pd.DataFrame], ...]:
    from DataManager import races_matrix_path, stages_matrix_path
    trained_model_path = get_models_path(params)
    stages_cyclist_matrix = import_data_from_csv(stages_matrix_path)
    races_cyclist_matrix = import_data_from_csv(races_matrix_path)
    model_name = params['model'][0]
    race_prediction = params['race_prediction'] if 'race_prediction' in params else False
    return model_name, race_prediction, races_cyclist_matrix, stages_cyclist_matrix, trained_model_path


def get_models_path(params: dict[str: Union[str, int, tuple[str, object]]]) -> str:
    from DataManager import get_data_path
    data_dir_path = get_data_path(params)
    models_dir_name = get_params_str({k: v for (k, v) in params.items() if k in MODELS_COLS})
    models_path = f'{data_dir_path}/{models_dir_name}'
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    return models_path


def append_baselines_results(data_files: str, exec_dir_path: str) -> None:
    result_df = pd.read_csv(data_files)
    for i in range(len(result_df)):
        append_row_to_csv(f'{exec_dir_path}/{FINAL_BASELINES_FILE_NAME}', result_df.iloc[i],
                          result_df.columns)


def append_model_results(data_files: str, exec_dir_path: str, raw_data_dir: str, data_dir: str, model_dir: str) -> None:
    results_path = f'{data_files}/{MODEL_RESULTS_FILE_NAME}'
    if os.path.exists(results_path):
        result_df = pd.read_csv(results_path)
        for i in range(len(result_df)):
            append_row_to_csv(f'{exec_dir_path}/{FINAL_MODEL_RESULTS_FILE_NAME}', result_df.iloc[i],
                              result_df.columns)
    else:
        error_row = {}
        raw_data_list = ast.literal_eval(raw_data_dir)
        for i in range(len(RAW_DATA_COLS)):
            error_row[RAW_DATA_COLS[i]] = raw_data_list[i]

        data_list = ast.literal_eval(data_dir)
        for i in range(len(DATA_COLS)):
            error_row[DATA_COLS[i]] = data_list[i]

        models_list = ast.literal_eval(model_dir)
        for i in range(len(MODELS_COLS)):
            error_row[MODELS_COLS[i]] = models_list[i]
        log(f'Problem to append results, params: {error_row}', 'ERROR', log_path=EXEC_PATH)
        append_row_to_csv(f'{exec_dir_path}/{ERROR_PARAMS_FILE_NAME}', error_row)


def append_results_from_files(exec_dir_path: str) -> None:
    for raw_data_dir in os.listdir(exec_dir_path):
        raw_data_dir_path = os.path.join(exec_dir_path, raw_data_dir)
        if os.path.isdir(raw_data_dir_path):
            for data_dir in os.listdir(raw_data_dir_path):
                data_dir_path = os.path.join(raw_data_dir_path, data_dir)
                if os.path.isdir(data_dir_path):
                    for file in os.listdir(data_dir_path):
                        data_files = os.path.join(data_dir_path, file)
                        if file == BASELINES_FILE_NAME:
                            append_baselines_results(data_files, exec_dir_path)
                        elif os.path.isdir(data_files):
                            model_dir = file
                            append_model_results(data_files, exec_dir_path, raw_data_dir, data_dir, model_dir)


def fit_clustering_algorithm(clustering_algorithm: Callable, k_clusters: int) -> None:
    from DataManager import filtered_stages_till_2016
    non_null_pred = (~filtered_stages_till_2016['distance'].isna()) & (
        ~filtered_stages_till_2016['elevation_gain'].isna())
    filtered_stages = filtered_stages_till_2016[non_null_pred]
    c_model = clustering_algorithm(n_clusters=k_clusters, random_state=0)
    c_model.fit_predict(filtered_stages[CLUSTERS_FEATURES])
    joblib.dump(c_model, f'{EXEC_PATH}/{CLUSTERING_ALG_NAME}_model_{k_clusters}.joblib')


def load_clusters(k_clusters: int):
    return joblib.load(f'{EXEC_PATH}/{CLUSTERING_ALG_NAME}_model_{k_clusters}.joblib')
