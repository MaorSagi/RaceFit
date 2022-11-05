APPEND_RESULTS, RUN_EXPERIMENTS = "append_results", "run_experiments"
EXPR_TASK = RUN_EXPERIMENTS
EXEC_NAME = 'Oct 30 - rewrite exprs'
EXECS_DIR_PATH = 'executions'
EXEC_PATH = f'{EXECS_DIR_PATH}/{EXEC_NAME}'
ALLOCATION_MATRICES_PATH = 'allocation_matrices'
SIMILARITY_COEFF = 10000000000000000.0
NUM_OF_POINTS = 40
K_POINTS = 10
DECISION_TIME_GAP = 1
WITHOUT_MIN_MAX_COLS = True

MODEL_RESULTS_FILE_NAME, FINAL_MODEL_RESULTS_FILE_NAME = 'ModelResults.csv', 'Final_Model_Results.csv'
BASELINES_FILE_NAME, FINAL_BASELINES_FILE_NAME = 'BaselinesResults.csv', 'Final_Baselines_Results.csv'
ERROR_PARAMS_FILE_NAME = 'ERROR_params.csv'

TP_SRC = 'TP'
STRAVA_SRC = 'STRAVA'
evaluate_baselines = ["cyclist_popularity_ranking_in_team", "cyclist_popularity_ranking_in_continent_in_team"]
RACE_CATEGORICAL = ['race_id', 'classification', 'nation', 'continent', 'race_date']

TIME_TRIAL_TYPES = ['Prologue', "Time trial", "Team Time Trial", "Individual Time Trial"]
difficulty_level_dict = {'EASY': 1, 'INTERMEDIATE': 2, 'HARD': 3}

GC, CLIMBER, TIME_TRIAL, SPRINT, ONE_DAY_RACES = "GC", "Climber", "Time Trial", "Sprint", "One Day Races"

parcours_type_name_dict = {'Flat': 1, 'Hills, flat finish': 2, 'Hills, uphill finish': 3, 'Mountains, flat finish': 4,
                           'Mountains, uphill finish': 5}

classification_dict = {'1.1': 1, '2.1': 2, '1.HC': 3, '2.HC': 4, '1.Pro': 5, '2.Pro': 6, '1.UWT': 7, '2.UWT': 8,
                       '1.WWT': 7, '2.WWT': 8, 'WC': 9}

IMPORTANT_RACES = ["La Vuelta ciclista a Espana", "Giro d'Italia", "Tour de France", "World Championships",
                   "Milano-Sanremo",
                   "Amstel Gold Race", "Tirreno-Adriatico", "Liege - Bastogne - Liege", "Liege-Bastogne-Liege",
                   "Il Lombardia", "La Fleche Wallonne"
    , "Paris-Roubaix", "Paris - Roubaix", "Paris - Nice", "Volta Ciclista a Catalunya", "Criterium du Dauphine",
                   "Tour des Flandres", "Gent-Wevelgem in Flanders Fields",
                   "Gent - Wevelgem in Flanders Fields", "Gent-Wevelgem In Flanders Fields",
                   "Donostia San Sebastian Klasikoa", "Clasica Ciclista San Sebastian"]

CYCLIST_GENERAL_COLS = ['cyclist_id', 'pcs_weight', 'pcs_height', 'age']  # 'cyclist_class', 'nation'

STAGES_COLS = ['stage_id', 'race_id', 'participated', 'race_name', 'race_date', 'race_year', 'stage_date', 'stage_name',
               'stage_number'
    , 'stage_type', 'continent', 'race_link', 'stage_link', 'race_category', 'stage_points_scale'
    , 'stage_ranking', 'profile_score', 'parcours_type', 'parcours_type_name', 'pcs_city_start',
               'pcs_city_finish', 'distance', 'elevation_gain',
               'temp_avg', 'nation', 'classification',
               'num_of_stages_in_race']  # 'elevation_maximum','elevation_minimum','elevation_loss','elevation_average','temp_max','temp_min','difficulty_level',

WORKOUTS_TP_COLS = ['cyclist_id', 'stage_id', 'last_modified_date', 'workout_tp_id', 'workout_type', 'workout_title',
                    'start_time', 'start_time_planned', 'completed', 'distance',
                    'distance_planned', 'distance_customized', 'total_time',
                    'total_time_planned', 'hr_min', 'hr_max', 'hr_avg', 'calories',
                    'calories_planned', 'TSS_actual', 'TSS_planned',
                    'TSS_calculation_method', 'IF', 'IF_planned', 'speed_avg',
                    'speed_planned', 'speed_max', 'norm_speed', 'norm_power', 'power_avg',
                    'power_max', 'energy', 'energy_planned', 'elevation_gain',
                    'elevation_gain_planned', 'elevation_loss', 'elevation_min',
                    'elevation_avg', 'elevation_max', 'torque_average', 'torque_maximum',
                    'temp_min', 'temp_avg', 'temp_max', 'cadence_avg', 'cadence_max',
                    'tags', 'tp_link', 'workout_datetime', 'workout_week', 'workout_month',
                    'workout_date']

WORKOUTS_STRAVA_COLS = ['cyclist_id', 'workout_strava_id', 'activity_type', 'workout_datetime',
                        'location', 'moving_time', 'speed_max', 'relative_effort',
                        'distance', 'elevation_gain', 'cadence_avg', 'temp_avg', 'hr_avg',
                        'intensity', 'historic_relative_effort', 'elapsed_time', 'calories',
                        'speed_avg', 'cadence_max', 'power_avg', 'massive_relative_effort',
                        'power_max', 'tough_relative_effort', 'energy', 'weighted_norm_power',
                        'device', 'training_load', 'hr_max',
                        'perceived_exertion', 'feels_like', 'humidity',
                        'weather', 'wind_direction', 'wind_speed', 'workout_title',
                        'workout_date', 'nation', 'continent']  # 'duration' , 'total_time', 'stage_id'

TP_CYCLIST_STATS_COLS = ['total_races_number_in_team', 'cyclist_total_races_number_in_team',
                         'cyclist_total_races_number_in_continent_in_team',
                         'cyclist_races_number_in_last_year', 'cyclist_races_rate_in_team',
                         'cyclist_races_rate_in_continent_in_team',
                         'total_races_number_in_continent_in_team',
                         'cyclist_popularity_ranking_in_team', 'cyclist_weeks_in_team',
                         'cyclist_popularity_ranking_in_continent_in_team',
                         'cyclist_weeks_since_last_race', 'distance_from_last_race', 'cyclist_gc_points',
                         'cyclist_one_day_races_points', 'cyclist_time_trial_points', 'cyclist_sprint_points',
                         'cyclist_climber_points']

STRAVA_CYCLIST_STATS_COLS = ['total_races_number_in_team', 'cyclist_total_races_number_in_team',
                             'cyclist_total_races_number_in_continent_in_team',
                             'total_races_number_in_continent_in_team',
                             'cyclist_races_number_in_last_year', 'cyclist_races_rate_in_team',
                             'cyclist_races_rate_in_continent_in_team',
                             'cyclist_popularity_ranking_in_team', 'cyclist_weeks_in_team',
                             'cyclist_popularity_ranking_in_continent_in_team',
                             'cyclist_weeks_since_last_race', 'distance_from_last_race', 'distance_from_last_workout',
                             'cyclist_gc_points',
                             'cyclist_one_day_races_points', 'cyclist_time_trial_points', 'cyclist_sprint_points',
                             'cyclist_climber_points'
                             ]

RAW_DATA_COLS = ['team_name', 'workouts_source', 'time_window', 'imputer_workouts']
DATA_COLS = ['col_threshold', 'imputer']  # 'row_threshold','scaler', 'aggregation_function'
MODELS_COLS = ['score_model', 'model', 'result_consideration']  # , 'feature_isolation'
LOG_LEVEL = 'create_input'
LOG_DICT = {'ERROR': 0, 'INFO': 1, 'Scores': 2, 'CV': 3, 'create_input': 4}
LEAVE_ONE_OUT = 'leave_one_out'
YEAR_SPLIT = 'year_split'
TO_FILTER_STAGES = ['race_date', 'stage_date', 'stage_name', 'stage_number', 'race_name', 'race_link',
                    'stage_link', 'race_category', 'pcs_city_start', 'pcs_city_finish', 'stage_points_scale',
                    'parcours_type', 'race_year', 'nation', 'start_time', 'stage_type', 'difficulty_level',
                    'parcours_type_name', 'race_oldest_pcs_link']
NON_AGG_RACE_COLS = set(TO_FILTER_STAGES) - {'race_year', 'nation', 'parcours_type', 'start_time'}

NON_AGG_CYCLIST_COLS = ['cyclist_id', 'workout_datetime']
START_YEAR_IN, END_YEAR_EX = 2017, 2023

TP_TO_FILTER_CYCLISTS = ['workout_date', 'tp_link', 'workout_week', 'workout_month', 'tags',
                         'start_time', 'last_modified_date', 'workout_tp_id', 'workout_type',
                         'workout_title', 'completed', 'total_races_number_in_team',
                         'total_races_number_in_continent_in_team']

STRAVA_TO_FILTER_CYCLISTS = ['workout_date', 'device', 'location', 'activity_type', 'workout_strava_id',
                             'total_races_number_in_team', 'total_races_number_in_continent_in_team', 'nation']

TP_HIGH_MISSING_RATE_WORKOUT_COLS = ['total_time_planned', 'TSS_planned', 'IF_planned', 'distance_planned',
                                     'speed_planned', 'distance_customized', 'calories_planned', 'norm_speed',
                                     'energy_planned', 'elevation_gain_planned', 'torque_average',
                                     'torque_maximum', 'TSS_calculation_method', 'stage_id', 'start_time_planned']

STRAVA_HIGH_MISSING_RATE_WORKOUT_COLS = ['historic_relative_effort', 'relative_effort', 'wind_direction', 'humidity',
                                         'duration', 'massive_relative_effort', 'tough_relative_effort', 'weather',
                                         'feels_like',
                                         'wind_speed', 'workout_title', 'perceived_exertion', 'stage_id']

TP_SUM_COLS = ['distance', 'total_time',
               'calories', 'energy', 'elevation_gain',
               'elevation_loss', 'TSS_actual']

TP_AVG_COLS = ['hr_min', 'hr_max', 'hr_avg'
    , 'IF', 'speed_avg', 'speed_max', 'norm_power', 'power_avg',
               'power_max', 'temp_min', 'temp_avg', 'temp_max', 'cadence_avg', 'cadence_max',
               'elevation_min', 'elevation_avg', 'elevation_max']
TP_LAST_COLS = ['workout_datetime']
