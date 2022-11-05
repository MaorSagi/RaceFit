import os.path
from typing import List, Dict
from utils import *
from expr_consts import *
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder

filtered_stages_results = pd.read_csv('./db/filtered_stages_results.csv')
stages_results = pd.read_csv('./db/stages_results.csv')
stages = pd.read_csv('./db/stages.csv')
filtered_stages = pd.read_csv('./db/filtered_stages.csv')
filtered_stages_till_2016 = pd.read_csv('./db/filtered_stages_till_2016.csv')
teams = pd.read_csv('./db/teams.csv')
cyclists_teams = pd.read_csv('./db/cyclists_teams.csv')
cyclists = pd.read_csv('./db/cyclists.csv')
cyclists_stats = pd.read_csv('./db/cyclists_stats.csv')
distances = pd.read_csv('./pre calculated data/nations_distances.csv')
teams_rankings = pd.read_csv('./pre calculated data/teams_rankings.csv')
workouts = pd.DataFrame()
workouts_cols: List[str]
team_names_dict = {}
team_pcs_to_id = {}
stages_matrix_path: str
races_matrix_path: str
cyclist_cols: List[str]
cyclist_stats_cols: List[str]
sum_cols: List[str]
avg_cols: List[str]
last_cols: List[str]
to_filter_cyclists_cols: List[str]
important_races_ids: List[int]
races_years_dict: Dict[int, int]

cols_to_encode = ['stage_type', 'continent', 'parcours_type_name', 'classification', 'nation',
                  'stage_nation', 'stage_continent']  # 'nation','difficulty_level',
important_races_lower = [r.lower() for r in IMPORTANT_RACES]


def init_data(team_id: int, workouts_source: str):
    global workouts, stages_matrix_path, races_matrix_path, important_races_ids, \
        races_years_dict, teams, team_names_dict, team_pcs_to_id,important_races_lower

    for t_id, g in teams.groupby('team_pcs_id'):
        team_names_dict[t_id] = g.loc[g['season'].idxmax()]['team_name']
        team_pcs_to_id[t_id] = list(g['team_id'].values)

    workouts = pd.read_csv(
        './db/tp_workouts.csv') if TP_SRC in workouts_source else pd.read_csv('./db/strava_workouts.csv')
    if team_id is not None:
        stages_matrix_path = f'allocation_matrices/stages_cyclist_matrix_{team_names_dict[team_id]}.csv'
        races_matrix_path = f'allocation_matrices/races_cyclist_matrix_{team_names_dict[team_id]}.csv'
    init_cols(workouts_source)
    init_cyclists_teams(team_id)
    init_cyclists()
    init_workouts()
    init_stages()
    init_stages_results(team_id)
    init_cyclists_stats()
    init_teams_rankings(team_id)

    important_races_ids = \
        stages[stages['race_name'].apply(lambda x: x.lower() in important_races_lower)][
            'race_id'].unique()
    races_years_dict = {r_id: r_date.year for r_id, r_date in
                        zip(stages['race_id'], stages['race_date'])}


def init_tp_workouts_cols():
    global workouts_cols, cyclist_stats_cols, sum_cols, avg_cols, last_cols, to_filter_cyclists_cols
    high_missing_rate_workout_cols = TP_HIGH_MISSING_RATE_WORKOUT_COLS
    to_filter_cyclists_cols = TP_TO_FILTER_CYCLISTS
    workouts_cols = [c for c in WORKOUTS_TP_COLS if
                     (c not in to_filter_cyclists_cols) and (c not in high_missing_rate_workout_cols)]
    if WITHOUT_MIN_MAX_COLS:
        min_max_cols = [c for c in workouts_cols if ("_max" in c) or ("_min" in c)]
        workouts_cols = [c for c in workouts_cols if c not in min_max_cols]
    cyclist_stats_cols = TP_CYCLIST_STATS_COLS
    sum_cols = TP_SUM_COLS
    avg_cols = TP_AVG_COLS
    last_cols = TP_LAST_COLS


def init_strava_workouts_cols():
    global workouts_cols, cyclist_stats_cols, sum_cols, avg_cols, last_cols, to_filter_cyclists_cols
    high_missing_rate_workout_cols = STRAVA_HIGH_MISSING_RATE_WORKOUT_COLS
    to_filter_cyclists_cols = STRAVA_TO_FILTER_CYCLISTS
    workouts_cols = [c for c in WORKOUTS_STRAVA_COLS if
                     (c not in to_filter_cyclists_cols) and (c not in high_missing_rate_workout_cols)]
    if WITHOUT_MIN_MAX_COLS:
        min_max_cols = [c for c in workouts_cols if ("_max" in c) or ("_min" in c)]
        workouts_cols = [c for c in workouts_cols if c not in min_max_cols]
    cyclist_stats_cols = STRAVA_CYCLIST_STATS_COLS
    cols_to_encode.extend(['nation', 'continent'])
    sum_cols = ['moving_time', 'elapsed_time', 'distance', 'elevation_gain', 'calories',
                'energy', 'total_time']
    avg_cols = ['intensity', 'training_load', 'hr_max', 'hr_avg', 'speed_avg', 'speed_max',
                'power_avg', 'weighted_norm_power', 'power_max', 'temp_avg', 'temp_max', 'cadence_avg',
                'cadence_max', 'elevation_loss', 'elevation_min', 'elevation_avg', 'elevation_max']
    last_cols = ['workout_datetime', 'nation', 'continent']


def init_cols(workouts_source):
    global workouts_cols, cyclist_cols
    if TP_SRC in workouts_source:
        init_tp_workouts_cols()
    else:
        init_strava_workouts_cols()
    workouts_cols = workouts_cols + ['number_of_workouts', 'number_of_workouts_above_28_deg']
    # workouts_cols_without_date = list(
    #     filter(lambda x: x not in ['workout_datetime', 'workout_date'], workouts_cols))
    cyclist_cols = CYCLIST_GENERAL_COLS + list(
        filter(lambda x: x != 'cyclist_id', workouts_cols)) + cyclist_stats_cols


def init_cyclists_teams(team_id):
    global cyclists_teams, workouts
    if team_id is not None:
        cyclists_teams = cyclists_teams[cyclists_teams['team_id'].isin(team_pcs_to_id[team_id])]
    workouts = workouts[workouts['cyclist_id'].isin(cyclists_teams['cyclist_id'])]
    for prop in ['start_date', 'stop_date']:
        cyclists_teams[prop] = pd.to_datetime(cyclists_teams[prop]).apply(lambda dt: dt.date())
    cyclists_teams['season'] = cyclists_teams['season'].apply(int)
    cyclists_teams = cyclists_teams.set_index('season', drop=False).sort_index()


def init_cyclists():
    global cyclists
    cyclists = cyclists[cyclists['cyclist_id'].isin(cyclists_teams['cyclist_id'])]


def init_workouts():
    global workouts
    workouts['workout_date'] = pd.to_datetime(workouts['workout_date']).apply(lambda dt: dt.date())
    workouts = workouts[workouts['workout_date'].apply(lambda dt: dt.year) >= START_YEAR_IN - 1]
    workouts = workouts[workouts['workout_date'].apply(lambda dt: dt.year) < END_YEAR_EX]


def init_stages():
    global stages, filtered_stages
    for prop in ['race_date', 'stage_date']:
        stages[prop] = pd.to_datetime(stages[prop]).apply(lambda dt: dt.date())
        filtered_stages[prop] = pd.to_datetime(filtered_stages[prop]).apply(lambda dt: dt.date())
    stages = stages[stages['race_date'].apply(lambda dt: dt.year) >= START_YEAR_IN]
    stages = stages[stages['race_date'].apply(lambda dt: dt.year) < END_YEAR_EX]
    filtered_stages = filtered_stages[filtered_stages['race_date'].apply(lambda dt: dt.year) < END_YEAR_EX]

    last_race_date = workouts['workout_date'].max()
    stages = stages[stages['race_date'] <= last_race_date]
    filtered_stages = filtered_stages[filtered_stages['race_date'] <= last_race_date]

    stages['start_time'] = stages['start_time'].apply(lambda t: float(str(t).split(' ')[0].replace(':', '')))
    filtered_stages['start_time'] = filtered_stages['start_time'].apply(
        lambda t: float(str(t).split(' ')[0].replace(':', '')))
    stages['parcours_type'] = stages['parcours_type'].apply(lambda t: t if t != 0 else None)
    filtered_stages['parcours_type'] = filtered_stages['parcours_type'].apply(lambda t: t if t != 0 else None)


def init_stages_results(team_id):
    global stages_results, filtered_stages_results
    if team_id is not None:
        stages_results = stages_results[stages_results['team_id'].isin(team_pcs_to_id[team_id])]
        filtered_stages_results = filtered_stages_results[
            filtered_stages_results['team_id'].isin(team_pcs_to_id[team_id])]
    stages_results = stages_results[stages_results['stage_id'].isin(stages['stage_id'].values)]
    filtered_stages_results = filtered_stages_results[
        filtered_stages_results['stage_id'].isin(filtered_stages['stage_id'].values)]
    filtered_stages_results = filtered_stages_results[
        filtered_stages_results['result_type'] != 'Teams classification']


def init_cyclists_stats():
    global cyclists_stats
    cyclists_stats['date'] = pd.to_datetime(cyclists_stats['date']).apply(lambda dt: dt.date())


def init_teams_rankings(team_id):
    global teams_rankings
    return teams_rankings[teams_rankings['team_pcs_id'] == team_id]


def get_raw_data_path(params):
    params_cpy = params.copy()
    team_id=params_cpy['team_id']
    if team_id is not None:
        params_cpy['team_name'] = team_names_dict[team_id]
    data_dir_name = get_params_str({k: v for (k, v) in params_cpy.items() if k in RAW_DATA_COLS})
    data_dir_path = f'{EXEC_PATH}/{data_dir_name}'
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)
    return data_dir_path


def get_data_path(params):
    raw_data_dir = get_raw_data_path(params)
    data_dir_name = get_params_str({k: v for (k, v) in params.items() if k in DATA_COLS})
    data_dir_path = f'{raw_data_dir}/{data_dir_name}'
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)
    return data_dir_path


def create_boolean_matrix():
    global filtered_stages_results, cyclists_teams, filtered_stages, stages_matrix_path, races_matrix_path
    # TODO - create matrix with non filtered stages and filter the stage later in the train
    stages_cyclists_df = (filtered_stages_results.merge(filtered_stages, on=['stage_id', 'race_id']))[
        ['cyclist_id', 'stage_id', 'race_id', 'race_date']]
    races_cyclists_df = stages_cyclists_df.copy()[['cyclist_id', 'race_id', 'race_date']].drop_duplicates()
    cyclists_teams_for_matrix = cyclists_teams.loc[
        (cyclists_teams['season'] >= START_YEAR_IN) & (cyclists_teams['season'] < END_YEAR_EX)].reset_index(drop=True)
    stages_cyclist_matrix = pd.get_dummies(stages_cyclists_df.set_index('stage_id')['cyclist_id']).max(level=0)
    races_cyclist_matrix = pd.get_dummies(races_cyclists_df.set_index('race_id')['cyclist_id']).max(level=0)

    cyclist_sorted = cyclists_teams_for_matrix.sort_values(['season', 'cyclist_id'], ascending=False)[
        'cyclist_id'].unique()
    cyclist_sorted = [float(c) for c in cyclist_sorted]
    missing_cyclists = list(set(cyclist_sorted) - set(stages_cyclist_matrix.columns))
    stages_cyclist_matrix[missing_cyclists] = 0
    missing_cyclists = list(set(cyclist_sorted) - set(races_cyclist_matrix.columns))
    races_cyclist_matrix[missing_cyclists] = 0

    stages_sorted = \
        pd.DataFrame(stages_cyclist_matrix.index, columns=['stage_id']).merge(stages_cyclists_df,
                                                                              on='stage_id').sort_values(
            ['race_date', 'race_id'],
            ascending=False)[
            'stage_id'].unique()
    stages_cyclist_matrix = stages_cyclist_matrix.loc[stages_sorted, cyclist_sorted].reset_index()
    stages_cyclist_matrix.to_csv(stages_matrix_path, index=False, header=True)

    race_sorted = \
        pd.DataFrame(races_cyclist_matrix.index, columns=['race_id']).merge(stages_cyclists_df,
                                                                            on='race_id').sort_values(
            'race_date',
            ascending=False)[
            'race_id'].unique()
    races_cyclist_matrix = races_cyclist_matrix.loc[race_sorted, cyclist_sorted].reset_index()
    races_cyclist_matrix.to_csv(races_matrix_path, index=False, header=True)


def preprocessing_cyclists_workout(params, cyclists_workouts):
    global sum_cols, cols_to_encode
    data_exec_path = get_raw_data_path(params)
    log('Start preprocessing_cyclists_workout()', 'create_input', log_path=data_exec_path)
    cw_to_fill = cyclists_workouts.loc[:, sum_cols]
    cols_to_drop = set(sum_cols) - set(cw_to_fill.columns)
    cyclists_workouts = cyclists_workouts.drop(columns=cols_to_drop)
    if ('imputer_workouts' in params) and (params['imputer_workouts'][0] != 'without'):
        imputer = params['imputer_workouts'][1]
        cw_to_fill = pd.DataFrame(imputer().fit_transform(cw_to_fill), index=cw_to_fill.index,
                                  columns=cw_to_fill.columns)
        cyclists_workouts.loc[:, cw_to_fill.columns] = cw_to_fill

    cols_to_encode = set.intersection(set(cols_to_encode), set(cyclists_workouts.columns))
    for col in cols_to_encode:
        if col in last_cols:
            continue
        else:
            raise ValueError(f'Unpredictable column {col}, in workouts preprocessing')
    return cyclists_workouts


def get_races_totals(races_cyclist_matrix):
    global filtered_stages
    race_total_distances = {}
    race_total_elevation_gains = {}
    num_of_stages_in_race = {}
    for i, r in races_cyclist_matrix.iterrows():
        race_group = filtered_stages[filtered_stages['race_id'] == r['race_id']]
        if race_group[race_group['distance'].isna()].empty:
            race_total_distances[r['race_id']] = race_group['distance'].sum()
        if race_group[race_group['elevation_gain'].isna()].empty:
            race_total_elevation_gains[r['race_id']] = race_group['elevation_gain'].sum()
        num_of_stages_in_race[r['race_id']] = len(race_group)
    return num_of_stages_in_race, race_total_distances, race_total_elevation_gains


def get_last_cyclists_records(X, Y, race_feature_key=None, race_key=None):
    X_filtered = X.copy()
    if race_feature_key:
        Y_filtered = Y[Y[race_feature_key] == race_key] if not Y.empty else Y
        X_filtered = X.loc[Y_filtered.index]
    last_cyclists_records = X_filtered.groupby('cyclist_id', as_index=False).last() if not X_filtered.empty else None
    return last_cyclists_records


def get_popularity_ranking_dict(cyclists_in_teams, last_cyclists_records, last_record_prop):
    cyclists_popularity_dict = {}
    for c in cyclists_in_teams['cyclist_id'].values:
        if last_cyclists_records is not None:
            cyclist_last_record = last_cyclists_records[last_cyclists_records['cyclist_id'] == float(c)]
            cyclists_popularity_dict[c] = cyclist_last_record.iloc[0][
                last_record_prop] if not cyclist_last_record.empty else 0
        else:
            cyclists_popularity_dict[c] = 0

    i = 1
    prev_popularity = -1
    cyclists_popularity_ranking_dict = {}
    cyclists_dict_for_rankings = sorted(cyclists_popularity_dict.items(), reverse=True, key=lambda item: item[1])
    for c, p in cyclists_dict_for_rankings:
        if (prev_popularity > -1) and (prev_popularity > p):
            i += 1
        cyclists_popularity_ranking_dict[c] = i
        prev_popularity = p
    return cyclists_popularity_ranking_dict


def get_cyclist_workouts_by_years(cyclists_workouts, years, end_time_window, cyclist_id):
    last_years_date = end_time_window - timedelta(days=years * 365)
    cyclist_w_last_years = cyclists_workouts[last_years_date:end_time_window]
    cyclist_w_last_years = cyclist_w_last_years[cyclist_w_last_years['cyclist_id'] == cyclist_id]
    return cyclist_w_last_years


def get_distance(nation_a, nation_b):
    if (str(nation_a) == 'nan') or (str(nation_b) == 'nan') or (nation_a is None) or (nation_b is None):
        return None
    if nation_a == nation_b:
        return 0
    pred_location = (distances['nation_A'] == nation_a) & (distances['nation_B'] == nation_b)
    if not distances.loc[pred_location].empty:
        distance = distances.loc[pred_location].iloc[0]['distance_AB']
        return distance
    else:
        raise ValueError(f'Distance missing between the nations - {nation_a},{nation_b}. ')


# currently return avg
def stages_aggregation_function(race):
    if isinstance(race, pd.DataFrame):
        res = dict()
        for c in race.columns:
            if c == 'parcours_type':
                res['is_flat'] = int(not race[race[c] == 1].empty)
                res['is_hills_flat'] = int(not race[race[c] == 2].empty)
                res['is_hills'] = int(not race[race[c] == 3].empty)
                res['is_mountains_flat'] = int(not race[race[c] == 4].empty)
                res['is_mountains_hills'] = int(not race[race[c] == 5].empty)
            elif c == 'stage_type':
                res['time_trial_stages_count'] = len(race[race[c].isin(TIME_TRIAL_TYPES)])
            elif c in RACE_CATEGORICAL:
                res[c] = race[c].iloc[0]
            elif c not in NON_AGG_RACE_COLS:
                res[c] = race[c].mean()
        res['num_of_stages_in_race'] = len(race)
    else:
        res = dict(race[(set(race.index) - set(NON_AGG_RACE_COLS)).union(set(RACE_CATEGORICAL))])
        res['is_flat'] = int(race['parcours_type'] == 1)
        res['is_hills_flat'] = int(race['parcours_type'] == 2)
        res['is_hills'] = int(race['parcours_type'] == 3)
        res['is_mountains_flat'] = int(race['parcours_type'] == 4)
        res['is_mountains_hills'] = int(race['parcours_type'] == 5)
        res['time_trial_stages_count'] = 0
        res['num_of_stages_in_race'] = 1
    return res


def create_input_data(params):
    from expr_consts import STAGES_COLS
    global cyclists, workouts, filtered_stages, cyclists_teams, cyclists_stats, \
        cyclist_cols, workouts_cols, sum_cols, stages_matrix_path, races_matrix_path
    stages_cyclist_matrix = import_data_from_csv(stages_matrix_path)
    races_cyclist_matrix = import_data_from_csv(races_matrix_path)
    import_input = params['import_input']
    overwrite = params['overwrite']
    race_prediction = params['race_prediction']
    X, Y = pd.DataFrame(), pd.DataFrame()
    data_exec_path = get_raw_data_path(params)
    if import_input and os.path.exists(f'{data_exec_path}/X_cols_raw_data.csv'):
        X = pd.read_csv(f'{data_exec_path}/X_cols_raw_data.csv')
        Y = pd.read_csv(f'{data_exec_path}/Y_cols_raw_data.csv')
    if overwrite:
        for file in os.listdir(data_exec_path):
            if os.path.isdir(f'{data_exec_path}/{file}'):
                for f in os.listdir(f'{data_exec_path}/{file}'):
                    if 'data' in file:
                        os.remove(f'{data_exec_path}/{file}/{f}')
            elif 'data' in file:
                os.remove(f'{data_exec_path}/{file}')

    time_window = params['time_window']
    id_feature = 'race_id' if race_prediction else 'stage_id'

    log('Start create_input_data()', 'create_input', log_path=data_exec_path)
    cyclists_general = cyclists.set_index('cyclist_id', drop=False).sort_index()
    stages_general = filtered_stages.set_index(id_feature, drop=False).sort_index()
    cyclists_workouts = workouts.set_index('workout_date', drop=False).sort_index()
    cyclists_workouts = preprocessing_cyclists_workout(params, cyclists_workouts)
    cyclists_start_date = {}
    for c, g in cyclists_teams.groupby('cyclist_id'):
        cyclists_start_date[c] = g.loc[g.sort_index().first_valid_index()]['start_date']
    if race_prediction:
        STAGES_COLS += ['time_trial_stages_count', 'is_flat', 'is_hills_flat', 'is_hills', 'is_mountains_flat',
                        'is_mountains_hills']
    else:
        STAGES_COLS += ['race_total_distance', 'race_total_elevation_gain']
        num_of_stages_in_race, race_total_distances, race_total_elevation_gains = get_races_totals(races_cyclist_matrix)
    input_matrix = races_cyclist_matrix if race_prediction else stages_cyclist_matrix
    for i, r in input_matrix[::-1].iterrows():
        id = r[id_feature]
        log(f"index: {len(input_matrix) - 1 - i}/{len(input_matrix) - 1},  id = {id}",
            'create_input', log_path=data_exec_path)
        r = r.drop(id_feature)
        race = stages_general.loc[id]
        race_continent = race['continent'] if isinstance(race, pd.Series) else race['continent'].iloc[0]
        race_date = race['race_date'] if isinstance(race, pd.Series) else race['race_date'].iloc[0]
        race_id = race['race_id'] if isinstance(race, pd.Series) else race['race_id'].iloc[0]

        till_race_cyclist_stats = cyclists_stats.loc[cyclists_stats['date'] < race_date]

        Y_prev = Y[Y['race_date'] < race_date] if not Y.empty else Y
        X_prev = X.loc[Y_prev.index] if not X.empty else X
        cyclists_in_teams = cyclists_teams.loc[cyclists_teams.index.intersection([race_date.year])]
        cyclists_in_teams = cyclists_in_teams.set_index('cyclist_id', drop=False).sort_index()

        last_cyclists_records = get_last_cyclists_records(X, Y)
        cyclists_popularity_ranking_dict = get_popularity_ranking_dict(cyclists_in_teams, last_cyclists_records,
                                                                       'cyclist_races_rate_in_team')

        last_cyclists_records_in_continent = get_last_cyclists_records(X, Y, 'continent', race_continent)
        cyclists_continent_popularity_ranking_dict = get_popularity_ranking_dict(cyclists_in_teams,
                                                                                 last_cyclists_records_in_continent,
                                                                                 'cyclist_races_rate_in_continent_in_team')

        for j, cyclist in cyclists_in_teams.iterrows():
            cyclist_id = cyclist['cyclist_id']
            cyclist_in_team = cyclists_in_teams.loc[cyclist_id]
            cyclist_race_stats = till_race_cyclist_stats[till_race_cyclist_stats['cyclist_id'] == cyclist_id]

            if (cyclist_in_team['start_date'] > race_date) or (race_date > cyclist_in_team['stop_date']):
                continue
            if not X.empty:
                c_events = X.loc[X['cyclist_id'] == cyclist_id]
                events_indices = c_events.index
                if (len(events_indices) > 0) and (not Y.loc[events_indices].loc[Y[id_feature] == id].empty):
                    continue
            log(f"cyclist: {cyclist_id}", 'create_input', debug=False, log_path=data_exec_path)
            cyclist_general = cyclists_general.loc[cyclists_general.index.intersection([cyclist_id])]

            end_time = race_date - timedelta(weeks=DECISION_TIME_GAP)
            if time_window:
                start_time = end_time - timedelta(weeks=time_window)
                cyclist_workouts = cyclists_workouts[start_time:end_time]
            else:
                cyclist_workouts = cyclists_workouts[:end_time]
            cyclist_workouts = cyclist_workouts[cyclist_workouts['cyclist_id'] == cyclist_id]
            cyclist_workouts.loc[:, 'stage_id'] = cyclist_workouts['stage_id'].apply(
                lambda s: 1 if str(s) != 'nan' else 0)
            cyclist_birthdate = datetime.strptime(cyclist_general.iloc[0]['date_of_birth'],
                                                  '%Y-%m-%d').date() if str(
                cyclist_general.iloc[0]['date_of_birth']) != 'nan' else None
            cyclist_general.at[
                cyclist_id, 'age'] = (race_date.year - cyclist_birthdate.year) if cyclist_birthdate else None
            cyclist_general_record = cyclist_general[CYCLIST_GENERAL_COLS]
            cyclist_record = cyclist_general_record.iloc[0].to_dict()
            cyclists_workouts_columns = set(cyclist_workouts.columns).intersection(workouts_cols)
            cyclists_workouts_columns = [c for c in cyclists_workouts_columns if
                                         c not in ['cyclist_id'] + list(cols_to_encode)]
            # CYCLIST_COLS = list(filter(lambda x: (x in NON_AGG_CYCLIST_COLS) or (x not in WORKOUTS_COLS), CYCLIST_COLS))
            for k in cyclists_workouts_columns:
                if (k != 'workout_datetime') and (f"{k}_last_year_avg_bias" not in cyclist_cols):
                    cyclist_cols.append(f"{k}_last_year_avg_bias")

            cyclist_workouts_last_year = get_cyclist_workouts_by_years(cyclists_workouts, 1, end_time, cyclist_id)
            cyclist_workouts_last_year = cyclist_workouts_last_year[cyclists_workouts_columns]
            cyclist_workouts_last_year = cyclist_workouts_last_year.drop(columns=['workout_datetime'])
            cyclist_workouts_last_year_mean = cyclist_workouts_last_year.mean()

            try:
                # CYCLIST_COLS = list(filter(lambda x: (x in NON_AGG_CYCLIST_COLS) or (x not in WORKOUTS_COLS), CYCLIST_COLS))
                # for window in range(1, max_time_window + 1):
                #     if window == 1:
                #         suffix = "_1_week_before_decision"
                #     else:
                #         suffix = f"_{window}_weeks_before_decision"
                #     CYCLIST_COLS = CYCLIST_COLS + [f"{c}{suffix}" for c in list(
                #         filter(lambda x: (x not in NON_AGG_CYCLIST_COLS) and ('_best_' not in x), WORKOUTS_COLS))]
                try:
                    # start_time = end_time - timedelta(weeks=window)
                    # window_cyclist_workouts = cyclist_workouts[start_time:end_time]
                    window_cyclist_workouts = cyclist_workouts
                    if window_cyclist_workouts.empty:
                        # cyclist_workouts_summary = {f"{k}{suffix}": None for k in cyclists_workouts_columns}
                        cyclist_workouts_summary = {k: None for k in cyclists_workouts_columns + last_cols}
                        cyclist_workouts_summary.update(
                            {f"{k}_last_year_avg_bias": None for k in cyclists_workouts_columns if
                             k != 'workout_datetime'})
                    else:
                        # cyclist_workouts_summary_without_date = cyclist_workouts[
                        #     workouts_cols_without_date]
                        # aggregation_function = params['aggregation_function'][1]
                        # cyclist_workouts_summary_without_date = aggregation_function(
                        #     cyclist_workouts_summary_without_date)
                        # cyclist_workouts_summary = cyclist_workouts_summary_without_date.copy()
                        # cyclist_workouts_summary['workout_datetime'] = cyclist_workouts[
                        #     'workout_datetime'].max()
                        # cyclist_workouts_summary['workout_date'] = cyclist_workouts[
                        #     'workout_date'].max()
                        aggregation_function = params['aggregation_function'][1]
                        cyclist_workouts_summary = aggregation_function(
                            window_cyclist_workouts[set(cyclists_workouts_columns + last_cols)])
                        workouts_window_mean = window_cyclist_workouts[cyclists_workouts_columns].mean()
                        for k in cyclists_workouts_columns:
                            if k != 'workout_datetime':
                                cyclist_workouts_summary[f"{k}_last_year_avg_bias"] = workouts_window_mean[k] - \
                                                                                      cyclist_workouts_last_year_mean[k]

                        # cyclist_workouts_summary = {f"{k}{suffix}": v for k, v in cyclist_workouts_summary.items()}
                    # CYCLIST_COLS = CYCLIST_COLS + list(cyclist_workouts_summary.keys())
                    cyclist_record.update(cyclist_workouts_summary)

                except:
                    log(f"Cyclist summary workouts error. Params: {get_params_str(params)}", "ERROR",
                        log_path=data_exec_path)
                cyclist_record = {k: (None if (k not in cyclist_record) else cyclist_record[k]
                                      ) for k in cyclist_cols}
                X_c = X_prev.loc[X_prev['cyclist_id'] == cyclist_id] if not X_prev.empty else X_prev
                Y_c = Y_prev.loc[X_c.index] if not Y_prev.empty else Y_prev
                cyclist_record['total_races_number_in_team'] = len(
                    Y_prev['race_id'].unique()) if not Y_prev.empty else 0
                cyclist_record['total_races_number_in_continent_in_team'] = len(
                    Y_prev[Y_prev['continent'] == race_continent]['race_id'].unique()) if not Y_prev.empty else 0
                cyclist_record['cyclist_total_races_number_in_team'] = len(
                    Y_c[Y_c['participated'] == 1]['race_id'].unique()) if not Y_c.empty else 0
                # TO DO : remove or keep??
                cyclist_record['cyclist_total_races_number_in_continent_in_team'] = len(
                    Y_c[(Y_c['participated'] == 1) & (Y_c['continent'] == race_continent)][
                        'race_id'].unique()) if not Y_c.empty else 0
                cyclist_last_year_pred = ((Y_c['participated'] == 1) & (
                        Y_c['race_date'] >= (race_date - relativedelta(years=1)))) if not Y_c.empty else None
                cyclist_record['cyclist_races_number_in_last_year'] = len(
                    Y_c[cyclist_last_year_pred]['race_id'].unique()) if not Y_c.empty else 0
                cyclist_record['cyclist_races_rate_in_team'] = (cyclist_record['cyclist_total_races_number_in_team'] /
                                                                cyclist_record['total_races_number_in_team']) \
                    if not Y_prev.empty else 0
                cyclist_record['cyclist_races_rate_in_continent_in_team'] = (
                        cyclist_record['cyclist_total_races_number_in_continent_in_team'] /
                        cyclist_record['total_races_number_in_continent_in_team']) \
                    if cyclist_record['total_races_number_in_continent_in_team'] > 0 else 0

                cyclist_record['cyclist_popularity_ranking_in_team'] = cyclists_popularity_ranking_dict[cyclist_id]
                cyclist_record['cyclist_popularity_ranking_in_continent_in_team'] = \
                    cyclists_continent_popularity_ranking_dict[cyclist_id]
                cyclist_weeks_in_team = (race_date - cyclists_start_date[cyclist_id]).days / 7
                cyclist_record['cyclist_weeks_in_team'] = cyclist_weeks_in_team
                last_race_location = None
                if not Y.empty:
                    cyclist_races = Y[(X['cyclist_id'] == cyclist_id) & (Y['participated'] == 1)]
                    if not cyclist_races.empty:
                        last_race = \
                            Y[(X['cyclist_id'] == cyclist_id) & (Y['participated'] == 1)].sort_values('race_date',
                                                                                                      ascending=False).iloc[
                                0]
                        last_race_date = last_race['race_date']
                        last_race_location = last_race['nation']
                        cyclist_record['cyclist_weeks_since_last_race'] = (race_date - last_race_date).days / 7
                    else:
                        cyclist_record['cyclist_weeks_since_last_race'] = None

                else:
                    cyclist_record['cyclist_weeks_since_last_race'] = cyclist_weeks_in_team

                race_location = race['nation'] if isinstance(race, pd.Series) else race['nation'].iloc[0]
                if 'nation' in X.columns:
                    if 'nation' in cyclist_record:
                        last_workout_location = cyclist_record['nation']
                        distance = get_distance(last_workout_location, race_location)
                        cyclist_record['distance_from_last_workout'] = distance
                    else:
                        cyclist_record['distance_from_last_workout'] = None
                if last_race_location is not None:
                    distance = get_distance(last_race_location, race_location)
                    cyclist_record['distance_from_last_race'] = distance
                else:
                    cyclist_record['distance_from_last_race'] = None

                def sum_speciality_points(stats_df, speciality_type):
                    return stats_df[stats_df['speciality_type'] == speciality_type]['points'].sum()

                cyclist_record['cyclist_gc_points'] = sum_speciality_points(cyclist_race_stats, GC)
                cyclist_record['cyclist_one_day_races_points'] = sum_speciality_points(cyclist_race_stats,
                                                                                       ONE_DAY_RACES)
                cyclist_record['cyclist_time_trial_points'] = sum_speciality_points(cyclist_race_stats, TIME_TRIAL)
                cyclist_record['cyclist_sprint_points'] = sum_speciality_points(cyclist_race_stats, SPRINT)
                cyclist_record['cyclist_climber_points'] = sum_speciality_points(cyclist_race_stats, CLIMBER)

                X = pd.concat([X, pd.DataFrame([cyclist_record])], ignore_index=True)
                append_row_to_csv(f'{data_exec_path}/X_cols_raw_data.csv', cyclist_record, cyclist_cols)
                if race_prediction:
                    event_record = stages_aggregation_function(race)
                    # TODO: add features to stages cols, something like the bias features added
                    # for k in cyclists_workouts_columns:
                    #     if (k != 'workout_datetime') and (f"{k}_last_year_avg_bias" not in CYCLIST_COLS):
                    #         CYCLIST_COLS.append(f"{k}_last_year_avg_bias")
                else:
                    event_record = dict(stages_general.loc[id])
                    event_record['race_total_distance'] = race_total_distances[
                        race_id] if race_id in race_total_distances else None
                    event_record['race_total_elevation_gain'] = race_total_elevation_gains[
                        race_id] if race_id in race_total_elevation_gains else None
                    event_record['num_of_stages_in_race'] = num_of_stages_in_race[race_id]
                event_record['race_year'] = race_date.year

                cyclists_in_event = [float(x) for x in list(r[r == 1].index)]
                if float(cyclist_id) in cyclists_in_event:
                    event_record['participated'] = 1
                else:
                    event_record['participated'] = 0
                event_record = {k: (None if (k not in event_record) else event_record[k]
                                    ) for k in STAGES_COLS}
                Y = pd.concat([Y, pd.DataFrame([event_record])], ignore_index=True)
                append_row_to_csv(f'{data_exec_path}/Y_cols_raw_data.csv', event_record, STAGES_COLS)
            except:
                log(f"X and Y record appending error. Params: {get_params_str(params)}", "ERROR",
                    log_path=data_exec_path)
    # X.to_csv(f'{data_exec_path}/X_cols_raw_data.csv', header=True, index=False)
    # Y.to_csv(f'{data_exec_path}/Y_cols_raw_data.csv', header=True, index=False)


def data_cleaning_and_preprocessing(cyclists, stages, params):
    data_exec_path = get_data_path(params)
    y = stages['participated']
    cyclists_datetime_cols = [c for c in cyclists.columns if 'workout_datetime' in c]
    cyclists = cyclists.drop(
        columns=cyclists_datetime_cols + list(set(to_filter_cyclists_cols).intersection(cyclists.columns)))
    stages = stages.drop(columns=list(set(TO_FILTER_STAGES).intersection(stages.columns)))
    stages = stages.rename(
        columns=dict(temp_avg='stage_temp_avg', distance='stage_distance', elevation_gain='stage_elevation_gain',
                     nation='stage_nation', continent='stage_continent'))
    X = pd.concat([cyclists, stages], axis=1).drop(columns='participated')
    X.to_csv(f'{data_exec_path}/X_cols_data_after_cleaning.csv', index=False, header=True)
    y.to_csv(f'{data_exec_path}/y_cols_data_after_cleaning.csv', index=False, header=True)

    if 'col_threshold' in params:
        if params['col_threshold'] != 'without':
            X = X.dropna(axis='columns', thresh=params['col_threshold'] * len(X.index))
        else:
            X = X.dropna(axis='columns', thresh=0.02 * len(X.index))
    if ('row_threshold' in params) and (params['row_threshold'] != 'without') and (params['row_threshold'] is not None):
        X = X.dropna(axis='index', thresh=params['row_threshold'] * len(X.columns))
        y = y.loc[X.index]

    X.to_csv(f'{data_exec_path}/X_cols_data_after_thresholds.csv', index=False, header=True)
    y.to_csv(f'{data_exec_path}/y_cols_data_after_thresholds.csv', index=False, header=True)

    X_cols_to_encode = set.intersection(set(cols_to_encode), set(X.columns))
    for col in X_cols_to_encode:
        if col == 'parcours_type_name':
            X[col] = X[col].map(parcours_type_name_dict)
        elif col == 'difficulty_level':
            X[col] = X[col].map(difficulty_level_dict)
        elif col == 'classification':
            X[col] = X[col].map(classification_dict)
        else:
            X[col] = LabelEncoder().fit_transform(X[col])
    X.to_csv(f'{data_exec_path}/X_cols_data_after_encoder.csv', index=False, header=True)

    if ('scaler' in params) and (params['scaler'] is not None) and (params['scaler'][0] != 'without'):
        scaler = params['scaler'][1]
        X_ids = X[['cyclist_id', 'stage_id', 'race_id']]
        X_to_fit = X.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
        X = pd.DataFrame(scaler().fit_transform(X_to_fit), index=X.index, columns=X_to_fit.columns)
        X[['cyclist_id', 'stage_id', 'race_id']] = X_ids
        X.to_csv(f'{data_exec_path}/X_cols_data_after_scaler.csv', index=False, header=True)
    return X, y


def data_imputation(params, X):
    if ('imputer' in params) and (params['imputer'][0] != 'without'):
        imputer = params['imputer'][1]
        X = pd.DataFrame(imputer().fit_transform(X), index=X.index, columns=X.columns)
        data_exec_path = get_data_path(params)
        X.to_csv(f'{data_exec_path}/X_cols_data_after_imputation.csv', index=False, header=True)
    return X


def get_race_date_by_race(race_id):
    race_vector = stages.loc[stages['race_id'] == race_id]
    if race_vector.empty:
        raise ValueError('Race vector is missing')
    race_date = race_vector['race_date'].iloc[0]
    return race_date


def get_race_date_by_stage(stage_id):
    race_vectors = stages.loc[stages['stage_id'] == stage_id]
    if race_vectors.empty:
        raise ValueError('Race Stages vectors are missing')
    race_date = race_vectors['race_date'].iloc[0]
    return race_date


def get_teams_cyclists_in_year(year):
    return cyclists_teams.loc[cyclists_teams.index.intersection([year])]
