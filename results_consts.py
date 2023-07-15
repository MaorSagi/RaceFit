titles = {'precisions': 'Precision@i', 'recalls': 'Recall@i', 'recalls_kn': 'Recall@(n+k)'}
titles_short = {'precisions': 'pr', 'recalls_kn': 'r'}

params = {
    # 'row_threshold': 'Row Threshold',
    # 'imputer_workouts': 'Workouts Imputer Type',
    'col_threshold': 'Column Threshold',
    'imputer': 'Imputer Type', 'scaler': 'Scaler Type',
    'time_window': 'Time Window',
    'model': 'Model',
    'score_model': 'Score Model',
    # 'aggregation_function': 'Aggregation Function',
    # 'similarity': "Similarity Function",
    # "popularity_weight":"Popularity Weight",
    # "workouts_source":"Workouts Data Src"
}
MODELS_LIST = ['LGBM', 'XGBoost', 'CatBoost', 'RandomForest', 'GaussianNB', 'AdaBoost', 'KNN', 'DecisionTree',
               'Logistic', 'GradientBoosting', 'SVC']
RESULTS_LISTS = {'model': MODELS_LIST, 'col_threshold': [0.7, 0.6, 0.35],
                 'team_name': [2088, 1330, 2040, 1103, 1136, 1253, 1258, 1060, 2738, 1187]}
workout_src_dict = {"Training Peaks": "TP", "STRAVA": "STRAVA"}
TEAM_NAMES = {1060.0: 'AG2R Citroën Team', 1103.0: 'CCC Team', 1330.0: 'Team Jumbo-Visma',
              2088.0: 'Lotto Fix All', 2738.0: 'Israel - Premier Tech', 1136.0: 'Cofidis', 1187.0: 'Groupama - FDJ',
              2040.0: 'Movistar Team', 1253.0: 'UAE Team Emirates', 1258.0: 'Trek - Segafredo'}
TEAMS_RANKING_DICT = {'Team Jumbo-Visma': 1, 'UAE Team Emirates': 4, 'Groupama - FDJ': 8, 'Trek - Segafredo': 9,
                      'AG2R Citroën Team': 12,
                      'Cofidis': 15, 'Movistar Team': 18, 'Lotto Fix All': 19, 'Israel - Premier Tech': 20,
                      'CCC Team': None}
TEAMS_TO_IGNORE = {'Cofidis', 'UAE Team Emirates', 'Trek - Segafredo', 'Movistar Team', 'CCC Team',
                          'Lotto Fix All','AG2R Citroën Team'}

EXTRA_LIM_GAP = 0.008

K_POINTS = 10
num_of_points = 10


#    'cyclist_id',    'stage_id',    'race_id',
strava_final_features_before_dropping_thresholds = [
    'nation', 'continent',  # todo: to remove
    'pcs_weight',
    'pcs_height',
    'age',
    'moving_time',
    'power_avg',
    'speed_max',  # todo: to remove
    'total_time',  # todo: to remove
    'power_max',  # todo: to remove
    'training_load',
    'intensity',
    'weighted_norm_power',
    'distance',
    'elevation_gain',
    'cadence_avg',
    'temp_avg',
    'elapsed_time',
    'calories',
    'speed_avg',
    'cadence_max',  # todo: to remove
    'energy',
    'number_of_workouts',
    'number_of_workouts_above_28_deg',
    'distance_from_last_workout',
    'cyclist_total_races_number_in_team',
    'cyclist_total_races_number_in_continent_in_team',
    'cyclist_races_number_in_last_year',
    'cyclist_races_rate_in_team',
    'cyclist_races_rate_in_continent_in_team',
    'cyclist_popularity_ranking_in_team',
    'cyclist_weeks_in_team',
    'cyclist_popularity_ranking_in_continent_in_team',
    'cyclist_weeks_since_last_race',
    'distance_from_last_race',
    'time_trial_stages_count',
    'moving_time_last_year_avg_deviation',
    'energy_last_year_avg_deviation',
    'distance_last_year_avg_deviation',
    'cadence_max_last_year_avg_deviation',  # todo: to remove
    'training_load_last_year_avg_deviation',
    'intensity_last_year_avg_deviation',
    'hr_avg_last_year_avg_deviation',
    'elevation_gain_last_year_avg_deviation',
    'temp_avg_last_year_avg_deviation',
    'calories_last_year_avg_deviation',
    'elapsed_time_last_year_avg_deviation',
    'cadence_avg_last_year_avg_deviation',
    'speed_max_last_year_avg_deviation',  # todo: to remove
    'total_time_last_year_avg_deviation',  # todo: to remove
    'power_max_last_year_avg_deviation',  # todo: to remove
    'power_avg_last_year_avg_deviation',
    'weighted_norm_power_last_year_avg_deviation',
    'speed_avg_last_year_avg_deviation',
    'difficulty_level',
    'stage_continent',
    'stage_ranking',
    'profile_score',
    'parcours_type_name',
    'stage_distance',
    'stage_elevation_gain',
    'stage_temp_avg',
    'classification',
    'race_total_distance',
    'race_total_elevation_gain',
    'num_of_stages_in_race',
    'is_flat',
    'is_hills_flat',
    'is_hills',
    'is_mountains_hills',
    'is_mountains_flat',
    'cyclist_sprint_points',
    'cyclist_climber_points',
    'cyclist_one_day_races_points',
    'cyclist_gc_points',
    'cyclist_time_trial_points'
]

#    'cyclist_id',    'stage_id',    'race_id',
tp_final_features_before_dropping_thresholds = [
    'pcs_weight',
    'pcs_height',
    'age',
    'distance',
    'total_time',
    'hr_avg',
    'calories',
    'TSS_actual',
    'IF',
    'speed_avg',
    'norm_power',
    'power_avg',
    'energy',
    'elevation_gain',
    'elevation_loss',
    'elevation_avg',
    'temp_avg',
    'cadence_avg',
    'number_of_workouts',
    'number_of_workouts_above_28_deg',
    'cyclist_total_races_number_in_team',
    'cyclist_total_races_number_in_continent_in_team',
    'cyclist_races_number_in_last_year',
    'cyclist_races_rate_in_team',
    'cyclist_races_rate_in_continent_in_team',
    'cyclist_popularity_ranking_in_team',
    'cyclist_weeks_in_team',
    'cyclist_popularity_ranking_in_continent_in_team',
    'cyclist_weeks_since_last_race',
    'distance_from_last_race',
    # 'time_trial_stages_count',
    'hr_avg_last_year_avg_deviation',
    'calories_last_year_avg_deviation',
    'elevation_gain_last_year_avg_deviation',
    'power_avg_last_year_avg_deviation',
    'elevation_loss_last_year_avg_deviation',
    'elevation_avg_last_year_avg_deviation',
    'total_time_last_year_avg_deviation',
    'temp_avg_last_year_avg_deviation',
    'distance_last_year_avg_deviation',
    'TSS_actual_last_year_avg_deviation',
    'IF_last_year_avg_deviation',
    'speed_avg_last_year_avg_deviation',
    'cadence_avg_last_year_avg_deviation',
    'hr_avg_last_last_year_avg_deviation',
    'norm_power_last_year_avg_deviation',
    'energy_last_year_avg_deviation',
    'difficulty_level',
    'stage_continent',
    'stage_ranking',
    'profile_score',
    'parcours_type_name',
    'stage_distance',
    'stage_elevation_gain',
    'stage_temp_avg',
    'classification',
    'race_total_distance',
    'race_total_elevation_gain',
    'num_of_stages_in_race',
    'is_flat',
    'is_hills_flat',
    'is_hills',
    'is_mountains_hills',
    'is_mountains_flat',
    'cyclist_sprint_points',
    'cyclist_climber_points',
    'cyclist_one_day_races_points',
    'cyclist_gc_points',
    'cyclist_time_trial_points'
]

strava_final_features_before_dropping_thresholds_formal = [
    'Cyclist last nation', 'Cyclist last continent',  # todo: to remove
    'Cyclist weight',
    'Cyclist height',
    'Cyclist age',
    'Workout total moving time',
    'Workout avg power',
    'Workout avg speed maxes',  # todo: to remove
    'Workout total time',  # todo: to remove
    'Workout avg power maxes',  # todo: to remove
    'Workout avg training load',
    'Workout avg intensity',
    'Workout avg norm power',
    'Workout total distance',
    'Workout total elevation gain',
    'Workout avg cadence',
    'Workout avg temp',
    'Workout elapsed time',
    'Workout total calories',
    'Workout avg speed',
    'Workout avg cadence maxes',  # todo: to remove
    'Workout total energy',
    'Workout count',
    'Workout count > 28 deg',
    'Workout last location distance from upcoming race',
    'Cyclist total race count till upcoming race',
    'Cyclist total race count in race continent',
    'Cyclist total race count last year till upcoming race',
    'Cyclist race rate till upcoming race',
    'Cyclist race rate in continent till upcoming race',
    'Cyclist popularity ranking till upcoming race',
    'Cyclist week count on team till upcoming race',
    'Cyclist popularity ranking in continent till upcoming race',
    'Cyclist week count since raced till upcoming race',
    'Cyclist last race distance from upcoming race',
    'Race time trial stages count',
    'Workout avg moving time last year avg difference',
    'Workout avg energy last year avg difference',
    'Workout avg distance last year avg difference',
    'Workout avg cadence maxes last year avg difference',  # todo: to remove
    'Workout avg training load last year avg difference',
    'Workout avg intensity last year avg difference',
    'Workout avg hr last year avg difference',
    'Workout avg elevation gain last year avg difference',
    'Workout avg temp last year avg difference',
    'Workout avg calories last year avg difference',
    'Workout avg elapsed time last year avg difference',
    'Workout avg cadence last year avg difference',
    'Workout avg speed maxes last year avg difference',  # todo: to remove
    'Workout avg total time last year avg difference',  # todo: to remove
    'Workout avg power maxes last year avg difference',  # todo: to remove
    'Workout avg power last year avg difference',
    'Workout avg norm power last year avg difference',
    'Workout avg speed last year avg difference',
    'Race difficulty level',  # e.g., Hard, Intermediate, Easy
    'Race continent',
    'Stage ranking',
    'Stage profile score',
    'Stage profile type',  # i.e., hilly, mountains,flat...
    'Stage distance',
    'Stage elevation gain',
    'Stage temp avg',
    'Race classification',
    'Race total stages distance',
    'Race total stages elevation gain',
    'Race number of stages',
    'Race has flat profile',
    'Race has hills and flat finish',
    'Race has hilly profile',
    'Race has mountains and hilly finish',
    'Race has mountains flat finish',
    'Cyclist sprint points till upcoming race',
    'Cyclist climber points till upcoming race',
    'Cyclist one day race points till upcoming race',
    'Cyclist gc points till upcoming race',
    'Cyclist time trial points till upcoming race'
]

tp_final_features_before_dropping_thresholds_formal = [
    'Cyclist weight',
    'Cyclist height',
    'Cyclist age',
    'Workout total distance',
    'Workout total duration',
    'Workout avg hr',
    'Workout total calories',
    'Workout total TSS',
    'Workout avg IF',
    'Workout avg speed',
    'Workout avg norm power',
    'Workout avg power',
    'Workout total energy',
    'Workout total elevation gain',
    'Workout total elevation loss',
    'Workout avg elevation',
    'Workout avg temp',
    'Workout avg cadence',
    'Workout count',
    'Workout count > 28 deg',
    'Cyclist total race count till upcoming race',
    'Cyclist total race count in race continent',
    'Cyclist total race count last year till upcoming race',
    'Cyclist race rate till upcoming race',
    'Cyclist race rate in continent till upcoming race',
    'Cyclist popularity ranking till upcoming race',
    'Cyclist week count on team till upcoming race',
    'Cyclist popularity ranking in continent till upcoming race',
    'Cyclist week count since raced till upcoming race',
    'Cyclist last race distance from upcoming race',
    # 'time trial stages count',
    # how should I call this feature? the indicative word (first word) is cyclist
    'Workout avg hr last year avg difference',
    'Workout avg calories last year avg difference',
    'Workout avg elevation gain last year avg difference',
    'Workout avg power last year avg difference',
    'Workout avg elevation loss last year avg difference',
    'Workout avg elevation last year avg difference',
    'Workout avg duration last year avg difference',
    'Workout avg temp last year avg difference',
    'Workout avg distance last year avg difference',
    'Workout avg TSS last year avg difference',
    'Workout avg IF last year avg difference',
    'Workout avg speed last year avg difference',
    'Workout avg cadence last year avg difference',
    'Workout avg hr last year avg difference',
    'Workout avg norm power last year avg difference',
    'Workout avg energy last year avg difference',
    'Race difficulty level',  # e.g., Hard, Intermediate, Easy
    'Race continent',
    'Stage ranking',
    'Stage profile score',
    'Stage profile type',  # i.e., hilly, mountains,flat...
    'Stage distance',
    'Stage elevation gain',
    'Stage temp avg',
    'Race classification',  # i.e., WorldTour 1.WT
    'Race total stages distance',
    'Race total stages elevation gain',
    'Race number of stages',
    'Race has flat profile',
    'Race has hills and flat finish',
    'Race has hilly profile',
    'Race has mountains and hilly finish',
    'Race has mountains flat finish',
    'Cyclist sprint points till upcoming race',
    'Cyclist climber points till upcoming race',
    'Cyclist one day race points till upcoming race',
    'Cyclist gc points till upcoming race',
    'Cyclist time trial points till upcoming race'
]


tp_features_names_dict = {k: v for k, v in
                          zip(tp_final_features_before_dropping_thresholds,
                              tp_final_features_before_dropping_thresholds_formal)}
strava_features_names_dict = {k: v for k, v in
                              zip(strava_final_features_before_dropping_thresholds,
                                  strava_final_features_before_dropping_thresholds_formal)}

BAR_PLOT_COLORS_DICT = {'Race': '#1e90ff', 'Stage': '#1e90ff', 'Workout': '#7eb54e', 'Cyclist': '#f39C12'}

ONE_DAY_RACES, MAJOR_TOURS, GRAND_TOURS = "One Day Races", "Major Tours", "Grand Tours"
curve_factor_x = 'recalls'  # 'precision_recall_curve'
curve_factor_y = 'precisions'
EXEC_NAME = "Dec 24 - TP stage prediction"  #"Dec 24 - TP stage prediction"#"Dec 24 - TP race prediction"#"Dec 5 - stage prediction" # "Sep 6-rerun - J paper results"#"Dec 5 - race prediction"
EXEC_BASE_PATH = f"M:/Maor/Expr/maors_code/executions/{EXEC_NAME}"  #f"./executions/{EXEC_NAME}"#
WORKOUTS_SRC = 'TP'

agg_func = 'SmartAgg'
RACEFIT_CYCLIST_STAGE = "RaceFit - Stage-Cyclist"
POPULARITY_IN_CONTINENT = "Popularity in continent"
POPULARITY_IN_GENERAL = "Popularity in general"
POPULARITY_IN_RACE_CLASS = "Popularity in race class"  # classification
POPULARITY_IN_RACE_TYPE = "Popularity in race type"  # one day/stage race
baseline_algorithms = {"cyclist_popularity_ranking_in_team": POPULARITY_IN_GENERAL,
                       "cyclist_popularity_ranking_in_continent_in_team": POPULARITY_IN_CONTINENT,
                       "cyclist_popularity_ranking_in_race_class_in_team": POPULARITY_IN_RACE_CLASS,
                       "cyclist_popularity_ranking_in_race_type_in_team": POPULARITY_IN_RACE_TYPE,
                       RACEFIT_CYCLIST_STAGE: RACEFIT_CYCLIST_STAGE
                       }
features_names_dict = tp_features_names_dict if WORKOUTS_SRC == 'TP' else strava_features_names_dict
features_color_dict = {f: BAR_PLOT_COLORS_DICT[f.split()[0]] for f in features_names_dict.values()}
LINE_WIDTH = 5


WITHOUT_SCORE_MODEL = True
PLOT_ALL_TEAMS_AVG = False  # None
SINGLE_RACE_TYPE = None
SINGLE_TEAM = "Israel - Premier Tech"#"Groupama - FDJ"#"Israel - Premier Tech"  # "AG2R Citroën Team"  # None

MULTIPLE_FILES = False
ZOOM_IN = None#{'x':(0,1),'y':(0,1)}

PLOT_AUC_PR = False

AUC_PR_INTERACTION = False

PLOT_EXPR_RESULTS = True
WITH_BASELINE = True
WITH_MODEL_BASELINE=False

result_list = []  # [f"{EXEC_BASE_PATH}/['without', 5, 'STRAVA', 'Israel - Premier Tech']/[0.4, 'SimpleImputer']/['CatBoost']/ModelResults.csv"]  # [f"{EXEC_BASE_PATH}/[0.7, 0.7, 'SimpleImputer', 'without', 5, 'SmartAgg', 'TP', 'Israel - Premier Tech']/['CatBoost', None]/ModelResults.csv"]
baseline_results_path = f"{EXEC_BASE_PATH}/Final_Baselines_Results 0.csv"
model_results_path = f"{EXEC_BASE_PATH}/Final_Model_Results 0.csv"

PLOT_TIME = False
top_i = 8

PLOT_KN_RECALLS = False
xpoints = range(K_POINTS)

# global by stage without score
global_best_params_dict = {
    'col_threshold': '60%',  # None,#'0.5',
    'imputer': 'SimpleImputer',
    # 'imputer_workouts': 'without',
    # 'result_consideration':None,
    'time_window': 5,
    'model': 'CatBoost',
    # 'score_model': 'without',
    # 'score_model_split':None,
    # 'score_model':'CatBoost',
}


# # global by race
# global_best_params_dict = {
#     'col_threshold': '40%',  # None,#'0.5',
#     'imputer': 'SimpleImputer',
#     # 'imputer_workouts': 'without',
#     # 'result_consideration':None,
#     'time_window': 3,
#     'model': 'CatBoost',
#     # 'score_model': 'without',
#     # 'score_model_split':None,
#     # 'score_model':'CatBoost',
# }

# # global by stage with score
# global_best_params_dict = {
#     'col_threshold': '60%',  # None,#'0.5',
#     'imputer': 'SimpleImputer',
#     # 'imputer_workouts': 'without',
#     # 'result_consideration':None,
#     'time_window': 5,
#     'model': 'CatBoost',
#     'score_model': 'CatBoost',
#     'k_clusters':'C 3',
#     # 'score_model_split':None,
#     # 'score_model':'CatBoost',
# }

# # BY RACE
# team_best_params_dict = {'AG2R Citroën Team': {},
#                          'CCC Team': {},
#                          'Team Jumbo-Visma': {'time_window': 5},
#                          'Lotto Fix All': {},
#                          'Israel - Premier Tech': {'time_window': 7,'col_threshold': '60%'},
#                          'Cofidis': {},
#                          'Groupama - FDJ': {},
#                          'Movistar Team': {},
#                          'UAE Team Emirates': {},
#                          'Trek - Segafredo': {}
#                          }


# BY STAGE
team_best_params_dict = {'AG2R Citroën Team': {},
                         'CCC Team': {},
                         'Team Jumbo-Visma': {},
                         'Lotto Fix All': {},
                         'Israel - Premier Tech': {},
                         'Cofidis': {},
                         'Groupama - FDJ': {'col_threshold': '40%'},
                         'Movistar Team': {},
                         'UAE Team Emirates': {},
                         'Trek - Segafredo': {}
                         }

PLOT_ONLY_BEST = False  # 'model'
PLOT_ONLY_BEST_BY_TEAM = False

params_to_plot = {
                  'imputer': 'Imputer Type',
                  # 'imputer_workouts': 'Workouts Imputer Type',
                  # 'model': 'Model',
                  # 'score_model': 'Score Model',
                  # 'score_model_split':'Score model Split',
                  # 'k_clusters': 'Number of Clusters',
                  # 'result_consideration':'Results Consideration',
                  # 'row_threshold': 'Row Threshold',
                  # 'col_threshold': 'Column Threshold',
                  # 'time_window': 'Time Window Size',
                  #   'weighted_mean': 'Weighted Mean'
                  }
imputation_labels = {'without': 'Without imputation', 'SimpleImputer': 'With imputation',
                     'KNNImputer': 'With imputation – KNNImputer'}  # {'without': 'Without imputation', 'SimpleImputer': 'With imputation – SimpleImputer'}
model_labels = {'DecisionTree': 'Decision Tree', 'RandomForest': 'Random Forest', 'CatBoost': 'CatBoost',
                'Logistic': 'Logistic Regression', 'without': 'Without'}
models_to_plot = ['RandomForest', 'DecisionTree', 'CatBoost', 'Logistic']
# BEST_RUN_MODEL_DIR = f"['{best_params_dict['model']}', None]"


PLOT_FEATURE_IMPORTANCE = False
FI_IN_TABLE = False
FI_RANKING_TABLE = False

SHAP, IG, RF, CATBST, CHI = True, True, True, True, True
RFF = False
# BEST_MODEL_ITER = f"4_{best_params_dict['model']}.pkl"
random_state = 0
number_of_races_to_test_fi = 5
interaction_feature = None  # "distance_from_last_race"

PLOT_CATBOOST_TREE = False

PLOT_TREE = False
max_depth = 4
# tree_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/['DecisionTree', None]/0_DecisionTree.pkl"
# tree_img_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/['DecisionTree', None]"


# baseline_results_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/BaselinesResults.csv"
# model_results_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/{BEST_RUN_MODEL_DIR}/ModelResults.csv"
