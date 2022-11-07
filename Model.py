import json
import ast
import pickle
from utils import *


# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder


def leave_one_out(df, group_id):
    train_indices, test_indices = [], []
    df_grouped = df.groupby(group_id, sort=False)
    for idx, group in df_grouped:
        train_idx, test_idx = (set(df.index) - set(group.index)), set(group.index)
        train_indices.append(train_idx)
        test_indices.append(test_idx)
    return zip(train_indices, test_indices)


def grouped_time_series_split(df, group_id, cv_num_of_splits=None):
    number_of_groups = len(df[group_id].unique())
    if cv_num_of_splits is None:
        cv_num_of_splits = number_of_groups
    train_indices, test_indices = [], []
    # the num of stages in test not changes in TimeSeriesSplit so I did the same..
    if cv_num_of_splits == 1:
        num_of_groups_in_test = int(number_of_groups / 5)
    else:
        num_of_groups_in_test = int(number_of_groups / (cv_num_of_splits))
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


def data_split(split_type=None, *args):
    if split_type == YEAR_SPLIT:
        df, group_id, n = args
        df[group_id] = df['race_id'].map()
        res = grouped_time_series_split(df, group_id, n)
        return res
    if split_type == LEAVE_ONE_OUT:
        return leave_one_out(*args)
    return grouped_time_series_split(*args)


def miss_rate(df):
    miss_rate_dict = {}
    for col in df.columns:
        miss_rate_dict[col] = round(100 * (len(df[df[col].isna()]) / len(df)))
    return sorted(miss_rate_dict.items(), key=lambda item: item[1])


def get_team_ranking_bins_for_stage(stage_id: int) -> pd.DataFrame:
    from DataManager import teams_rankings
    return teams_rankings[teams_rankings['stage_id'] == stage_id]


def duplicate_high_results_records(X, y, times_to_mul_list_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_addition = pd.DataFrame(columns=X.columns)
    y_addition = pd.Series()
    times_to_mul_list: list[int, ...] = eval(times_to_mul_list_str)
    n_bins = len(times_to_mul_list)
    for s_id, g in X.groupby('stage_id'):
        team_ranking_for_stage = get_team_ranking_bins_for_stage(s_id)
        if team_ranking_for_stage.empty:
            continue
        teams_rankings_row = team_ranking_for_stage.iloc[0]
        cutoffs_str = teams_rankings_row[f'cutoffs_{n_bins}']
        if str(cutoffs_str) == 'nan':
            continue
        cutoffs = eval(cutoffs_str)
        ranking = teams_rankings_row[f'ranking']
        i = 0
        times_to_mul = get_times_to_multiply_the_record(cutoffs, i, ranking, times_to_mul_list)
        X_addition, y_addition = mul_records(X_addition, g, times_to_mul, y, y_addition)
    return X.append(X_addition, ignore_index=True), y.append(y_addition, ignore_index=True)


def mul_records(X_addition, g, times_to_mul, y, y_addition):
    for t in range(times_to_mul - 1):
        X_addition = X_addition.append(g, ignore_index=True)
        y_addition = y_addition.append(y[g.index], ignore_index=True)
    return X_addition, y_addition


def get_times_to_multiply_the_record(cutoffs, i, ranking, times_to_mul_list):
    times_to_mul = times_to_mul_list[-1]
    for c in cutoffs:
        if ranking <= c:
            times_to_mul = times_to_mul_list[i]
            break
        i += 1
    return times_to_mul


@time_wrapper
def train(iteration: int, params: dict[str: Union[str, int, tuple[str, object]]],
          cyclists: pd.DataFrame,
          stages: pd.DataFrame) -> None:
    trained_model_path = get_models_path(params)
    X, y, features_to_remove, model_name, overwrite, result_consideration = extract_training_parameters(
        cyclists, params, stages)
    try:
        model_filename = f"{iteration + 1}_{model_name}.pkl"
        model_full_path = f'{trained_model_path}/{model_filename}'
        if is_file_exists_and_should_not_be_changed(overwrite, model_full_path):
            return
        model = params['model'][1]['constructor']()
        if X.empty:
            return
        X_new, y_new = X.copy(), y.copy()
        if result_consideration:
            X_new, y_new = duplicate_high_results_records(X_new, y_new,result_consideration)
        features_to_remove = features_to_remove + ['cyclist_id', 'stage_id', 'race_id']
        X_new = X_new.drop(columns=set(X_new.columns).intersection(features_to_remove))
        train_func = params['model'][1]['train']
        train_func(model, X_new, y_new)
        pickle.dump(model, open(model_full_path, 'wb'))
    except Exception as err:
        print(f'{err}')
        print(get_params_str(params))
        model_exec_path = get_models_path(params)
        log(f"Params: {get_params_str(params)},Train : {err}\n {traceback.format_exc()}", "ERROR",
            log_path=model_exec_path)
        traceback.print_exc()
        print()


def extract_training_parameters(cyclists, params, stages):
    from DataManager import cyclist_stats_cols
    overwrite = params['overwrite']
    result_consideration = params['result_consideration']
    model_name = params['model'][0]
    y = stages['participated']
    X = pd.concat([cyclists, stages], axis=1).drop(columns='participated')
    feature_isolation = params['feature_isolation']
    if feature_isolation is not None:
        features_to_isolate = set(X.columns).intersection(cyclist_stats_cols + list(['race_total_distance',
                                                                                     'race_total_elevation_gain']))
        if feature_isolation == 'without_new_features':
            features_to_remove = features_to_isolate
        else:
            features_to_remove = list(filter(lambda f: f != feature_isolation, features_to_isolate))
    else:
        features_to_remove = []
    # features_to_remove = ['distance_from_last_workout', 'distance_from_last_race', 'cyclist_weeks_since_last_race']
    return X, y, features_to_remove, model_name, overwrite, result_consideration


def fill_races_pred_matrix(X_test, y_test, y_prob, races_cyclists_soft_pred_matrix):
    from DataManager import get_race_date_by_race, get_teams_cyclists_in_year
    for race_id, g in X_test.groupby('race_id'):
        race_date = get_race_date_by_race(race_id)
        cyclists = get_teams_cyclists_in_year(race_date.year)

        for i, r in g.iterrows():
            cyclist = r['cyclist_id']
            if cyclist in cyclists['cyclist_id'].values:
                cyclist_in_team = cyclists[cyclist == cyclists['cyclist_id']].iloc[0]
                if cyclist_in_team['start_date'] <= race_date <= cyclist_in_team['stop_date']:
                    cyclist_column = str(cyclist)
                    race_idx = \
                        races_cyclists_soft_pred_matrix[races_cyclists_soft_pred_matrix['race_id'] == race_id].index[
                            0]
                    pred_idx = y_test.index.get_loc(i)
                    races_cyclists_soft_pred_matrix.loc[race_idx, cyclist_column] = y_prob[pred_idx][1]
                    # .idxmax()
        return races_cyclists_soft_pred_matrix


def fill_stages_pred_matrix(X_test, y_test, y_prob, stages_cyclists_pred_matrix):
    from DataManager import get_race_date_by_stage, get_teams_cyclists_in_year
    stage_cyclists = {}
    for stage_id, g in X_test.groupby('stage_id'):
        race_date = get_race_date_by_stage(stage_id)
        cyclists = get_teams_cyclists_in_year(race_date.year)
        stage_cyclists[stage_id] = set()

        for i, r in g.iterrows():
            cyclist = r['cyclist_id']
            if cyclist in cyclists['cyclist_id'].values:
                cyclist_in_team = cyclists[cyclist == cyclists['cyclist_id']].iloc[0]
                if cyclist_in_team['start_date'] <= race_date <= cyclist_in_team['stop_date']:
                    cyclist_column = str(cyclist)
                    stage_cyclists[stage_id].add(cyclist_column)
                    stage_idx = \
                        stages_cyclists_pred_matrix[stages_cyclists_pred_matrix['stage_id'] == stage_id].index[
                            0]
                    pred_idx = y_test.index.get_loc(i)
                    stages_cyclists_pred_matrix.loc[stage_idx, cyclist_column] = y_prob[pred_idx][1]
                    # .idxmax()
        return stages_cyclists_pred_matrix, stage_cyclists


# prediction_evaluation_task
'''
Recall@k= (Relevant_Items_Recommended in top-k) / (Relevant_Items)
Precision@k= (Relevant_Items_Recommended in top-k) / (k_Items_Recommended)
'''


@time_wrapper
def evaluate(iteration, X_test, y_test, params):
    from DataManager import races_matrix_path, stages_matrix_path, cyclist_stats_cols
    trained_model_path = get_models_path(params)
    races_cyclist_matrix = import_data_from_csv(races_matrix_path)
    model_name = params['model'][0]
    race_prediction = params['race_prediction'] if 'race_prediction' in params else False

    if not race_prediction:
        stages_cyclist_matrix = import_data_from_csv(stages_matrix_path)
        stages_cyclists_cpy = stages_cyclist_matrix.copy()
        stages_cyclists_test_matrix = stages_cyclists_cpy.loc[
            stages_cyclists_cpy['stage_id'].isin(X_test['stage_id'])]
        stages_cyclists_test_matrix = stages_cyclists_test_matrix.reset_index(drop=True)
        missing_cyclists_in_test = list(
            filter(lambda c: str(c) not in stages_cyclists_test_matrix.columns, X_test['cyclist_id'].unique()))
        missing_cyclists_in_test = [str(c) for c in missing_cyclists_in_test]
        stages_cyclists_test_matrix[missing_cyclists_in_test] = 0

        stages_cyclists_pred_matrix = stages_cyclists_test_matrix.copy()
        stages_cyclists_pred_matrix[stages_cyclists_pred_matrix.columns[1:]] = None

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
    races_cyclists_pred_matrix = races_cyclists_pred_matrix.fillna(0)

    X_test_as_input = X_test.drop(columns=['stage_id', 'cyclist_id', 'race_id'])
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
    X_test_as_input = X_test_as_input.drop(columns=set(X_test_as_input.columns).intersection(features_to_remove))
    y_prob = None
    try:
        if os.path.exists(f'{trained_model_path}/{iteration}_{model_name}.pkl'):
            model = pickle.load(open(f'{trained_model_path}/{iteration}_{model_name}.pkl', 'rb'))
            # y_pred = params['model'][1]['predict'](model, X_test_as_input)
            y_prob = params['model'][1]['predict_proba'](model, X_test_as_input)
    except:
        log(f"Predict stage value. Params: {get_params_str(params)}", "ERROR",
            log_path=trained_model_path)

    try:
        if y_prob is not None:
            if race_prediction:
                races_cyclists_soft_pred_matrix = fill_races_pred_matrix(X_test, y_test, y_prob,
                                                                         races_cyclists_soft_pred_matrix)
            else:
                stages_cyclists_pred_matrix, stage_cyclists = fill_stages_pred_matrix(X_test, y_test, y_prob,
                                                                                      stages_cyclists_pred_matrix)

    except:
        log(f"Evaluate function: prediction job, params: {get_params_str(params)}", "ERROR",
            log_path=get_models_path(params))

    grouped_data_by_races = X_test.groupby('race_id')
    race_cyclists = {}
    i = 0
    total_scores = []
    for race_id, g in grouped_data_by_races:

        curr_race_pred = races_cyclists_test_matrix['race_id'] == race_id
        cyclists_to_choose = races_cyclists_test_matrix.loc[curr_race_pred, races_columns[1:]].sum(
            1)
        race_cyclists[race_id] = set([str(e) for e in X_test[X_test['race_id'] == race_id]['cyclist_id'].values])
        cyclists_in_team = race_cyclists[race_id]

        if race_prediction:
            cyclists_participated_in_race_predict = races_cyclists_soft_pred_matrix.loc[
                i, set.intersection(set(races_columns), cyclists_in_team)].fillna(0)
        else:
            stages_in_race = g['stage_id'].unique()
            curr_race_stages_pred = stages_cyclists_pred_matrix['stage_id'].isin(stages_in_race)
            cyclists_participated_in_race_predict = stages_cyclists_pred_matrix.loc[
                curr_race_stages_pred, set.intersection(set(races_columns), cyclists_in_team)]
            cyclists_participated_in_race_predict = cyclists_participated_in_race_predict.mean().fillna(0)

        if y_prob is None:
            top_cyclists = cyclists_participated_in_race_predict.sample(cyclists_to_choose.values[0]).index
        else:
            top_cyclists = cyclists_participated_in_race_predict.nlargest(cyclists_to_choose.values[0]).index

        # pred
        races_cyclists_pred_matrix.at[i, 'race_id'] = race_id
        races_cyclists_pred_matrix.loc[i, races_columns[1:]] = 0
        races_cyclists_pred_matrix.loc[i, top_cyclists] = 1
        # pred soft
        races_cyclists_soft_pred_matrix.at[i, 'race_id'] = race_id
        races_cyclists_soft_pred_matrix.loc[i, races_columns[1:]] = 0
        if y_prob is None:
            races_cyclists_soft_pred_matrix.loc[i, top_cyclists] = 1
        for c in cyclists_in_team:
            races_cyclists_soft_pred_matrix.at[i, c] = cyclists_participated_in_race_predict[c]

        i += 1
        prediction_matrix_path = f"{trained_model_path}/prediction_matrix.csv"
        total_scores = evaluate_results(params, iteration, race_cyclists, races_columns, races_cyclists_pred_matrix,
                                        races_cyclists_soft_pred_matrix, races_cyclists_test_matrix,
                                        prediction_matrix_path, log_path=trained_model_path)

    total_scores['precision_recall_curve'] = json.dumps(list(total_scores['precision_recall_curve']))
    total_scores['roc_curve'] = json.dumps(list(total_scores['roc_curve']))
    total_scores['precisions'] = json.dumps(list(total_scores['precisions']))
    total_scores['recalls'] = json.dumps(list(total_scores['recalls']))
    total_scores['recalls_kn'] = json.dumps(list(total_scores['recalls_kn']))

    # total_scores['norm_precisions'] = json.dumps(list(total_scores['norm_precisions']))
    # total_scores['norm_recalls'] = json.dumps(list(total_scores['norm_recalls']))

    '''
    import matplotlib.pyplot as plt
    plt.plot(recall_new_points,precision_new_points)
    plt.show()
    '''
    return total_scores


def get_models_path(params):
    from DataManager import get_data_path
    data_dir_path = get_data_path(params)
    models_dir_name = get_params_str({k: v for (k, v) in params.items() if k in MODELS_COLS})
    models_path = f'{data_dir_path}/{models_dir_name}'
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    return models_path


def append_baselines_results(data_files, exec_dir_path):
    result_df = pd.read_csv(data_files)
    for i in range(len(result_df)):
        append_row_to_csv(f'{exec_dir_path}/{FINAL_BASELINES_FILE_NAME}', result_df.iloc[i],
                          result_df.columns)


def append_model_results(data_files, exec_dir_path, raw_data_dir, data_dir, model_dir):
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


def fit_clustering_algorithm(clustering_algorithm_name, clustering_algorithm, k_clusters):
    from DataManager import filtered_stages_till_2016
    import joblib
    non_null_pred = (~filtered_stages_till_2016['distance'].isna()) & (
        ~filtered_stages_till_2016['elevation_gain'].isna())
    filtered_stages = filtered_stages_till_2016[non_null_pred]
    kmeans = clustering_algorithm(n_clusters=k_clusters, random_state=0)
    kmeans.fit_predict(filtered_stages[['distance', 'elevation_gain']])
    joblib.dump(kmeans, f'{EXEC_PATH}/{clustering_algorithm_name}_model_{k_clusters}.joblib')
