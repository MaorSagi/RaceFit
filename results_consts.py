titles = {'precisions': 'Precision@i', 'recalls': 'Recall@i', 'recalls_kn': 'Recall@(n+k)'}
params = {
    'row_threshold': 'Row Threshold',
    'imputer_workouts': 'Workouts Imputer Type',
    'col_threshold': 'Column Threshold',
    'imputer': 'Imputer Type', 'scaler': 'Scaler Type',
    'time_window': 'Time Window',
    'model': 'Model',
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
TEAMS_RANKING_DICT = {'Team Jumbo-Visma': 1,'UAE Team Emirates': 4,'Groupama - FDJ': 8, 'Trek - Segafredo': 9,'AG2R Citroën Team': 12,
                      'Cofidis': 15,'Movistar Team': 18, 'Lotto Fix All': 19,'Israel - Premier Tech': 20,'CCC Team': None}
MISSING_WORKOUTS_TEAMS = {'UAE Team Emirates', 'Trek - Segafredo','Movistar Team'}

EXTRA_LIM_GAP = 0.008

K_POINTS = 10
num_of_points = 10

#    'cyclist_id',    'stage_id',    'race_id',
strava_final_features_before_dropping_thresholds = [
    'nation','continent', # todo: to remove
    'pcs_weight',
    'pcs_height',
    'age',
    'moving_time',
    'power_avg',
    'speed_max',# todo: to remove
    'total_time',# todo: to remove
    'power_max',# todo: to remove
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
    'cadence_max',# todo: to remove
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
    'moving_time_last_year_avg_bias',
    'energy_last_year_avg_bias',
    'distance_last_year_avg_bias',
    'cadence_max_last_year_avg_bias',# todo: to remove
    'training_load_last_year_avg_bias',
    'intensity_last_year_avg_bias',
    'hr_avg_last_year_avg_bias',
    'elevation_gain_last_year_avg_bias',
    'temp_avg_last_year_avg_bias',
    'calories_last_year_avg_bias',
    'elapsed_time_last_year_avg_bias',
    'cadence_avg_last_year_avg_bias',
    'speed_max_last_year_avg_bias',# todo: to remove
    'total_time_last_year_avg_bias',# todo: to remove
    'power_max_last_year_avg_bias',# todo: to remove
    'power_avg_last_year_avg_bias',
    'weighted_norm_power_last_year_avg_bias',
    'speed_avg_last_year_avg_bias',
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
    'hr_avg_last_year_avg_bias',
    'calories_last_year_avg_bias',
    'elevation_gain_last_year_avg_bias',
    'power_avg_last_year_avg_bias',
    'elevation_loss_last_year_avg_bias',
    'elevation_avg_last_year_avg_bias',
    'total_time_last_year_avg_bias',
    'temp_avg_last_year_avg_bias',
    'distance_last_year_avg_bias',
    'TSS_actual_last_year_avg_bias',
    'IF_last_year_avg_bias',
    'speed_avg_last_year_avg_bias',
    'cadence_avg_last_year_avg_bias',
    'hr_avg last_last_year_avg_bias',
    'norm_power_last_year_avg_bias',
    'energy_last_year_avg_bias',
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
    'cyclist last nation','cyclist last continent', # todo: to remove
    'cyclist weight',
    'cyclist height',
    'cyclist age',
    'workouts \ntotal moving time',
    'workouts \navg power',
    'workouts \navg speed maxes', # todo: to remove
    'workouts \ntotal time',  # todo: to remove
    'workouts \navg power maxes',# todo: to remove
    'workouts \navg training load',
    'workouts \navg intensity',
    'workouts \navg norm power',
    'workouts \ntotal distance',
    'workouts total \nelevation gain',
    'workouts avg \ncadence',
    'workouts avg \ntemp',
    'workouts \nelapsed time',
    'workouts \ntotal calories',
    'workouts \navg speed',
    'workouts \navg cadence maxes', # todo: to remove
    'workouts \ntotal energy',
    'workouts count',
    'workouts count > 28 deg',
    'cyclist distance from \nlast workout',
    'cyclist total \nraces count',
    'cyclist total races \ncount in race continent',
    'cyclist total races \ncount last year',
    'cyclist races rate',
    'cyclist races \nrate in continent',
    'cyclist popularity \nranking',
    'cyclist weeks \ncount in team',
    'cyclist popularity \nranking in continent',
    'cyclist weeks count \nsince raced',
    'cyclist last race \ndistance from race',
    'race time trial \nstages count',
    'workouts avg moving time \nlast year avg deviation',
    'workouts avg energy \nlast year avg deviation',
    'workouts avg distance \nlast year avg deviation',
    'workouts avg cadence maxes \nlast year avg deviation', # todo: to remove
    'workouts avg training load \nlast year avg deviation',
    'workouts avg intensity \nlast year avg deviation',
    'workouts avg hr \nlast year avg deviation',
    'workouts avg elevation gain \nlast year avg deviation',
    'workouts avg temp \nlast year avg deviation',
    'workouts avg calories \nlast year avg deviation',
    'workouts avg elapsed time \nlast year avg deviation',
    'workouts avg cadence \nlast year avg deviation',
    'workouts avg speed maxes \nlast year avg deviation',# todo: to remove
    'workouts avg total time \nlast year avg deviation',# todo: to remove
    'workouts avg power maxes \nlast year avg deviation',# todo: to remove
    'workouts avg power \nlast year avg deviation',
    'workouts avg norm power \nlast year avg deviation',
    'workouts avg speed \nlast year avg deviation',
    'race difficulty level',  # e.g., Hard, Intermediate, Easy
    'race continent',
    'stage ranking',
    'stage profile score',
    'stage profile type',  # i.e., hilly, mountains,flat...
    'stage distance',
    'stage elevation gain',
    'stage temp avg',
    'race classification',
    'race total \nstages distance',
    'race total stages \nelevation gain',
    'race number \nof stages',
    'race has flat profile',
    'race has hills and flat finish',
    'race has hilly profile',
    'race has mountains and hilly finish',
    'race has mountains flat finish',
    'cyclist sprint points',
    'cyclist climber points',
    'cyclist one day \nraces points',
    'cyclist gc points',
    'cyclist time trial points'
]

tp_final_features_before_dropping_thresholds_formal = [
    'cyclist weight',
    'cyclist height',
    'cyclist age',
    'workouts \ntotal distance',
    'workouts \ntotal duration',
    'workouts \navg hr',
    'workouts \ntotal calories',
    'workouts \ntotal TSS',
    'workouts \navg IF',
    'workouts \navg speed',
    'workouts \navg norm power',
    'workouts \navg power',
    'workouts \ntotal energy',
    'workouts total \nelevation gain',
    'workouts total \nelevation loss',
    'workouts avg \nelevation',
    'workouts avg \ntemp',
    'workouts avg \ncadence',
    'workouts count',
    'workouts count > 28 deg',
    'cyclist total \nraces count',
    'cyclist total races \ncount in race continent',
    'cyclist total races \ncount last year',
    'cyclist races rate',
    'cyclist races \nrate in continent',
    'cyclist popularity \nranking',
    'cyclist weeks \ncount in team',
    'cyclist popularity \nranking in continent',
    'cyclist weeks count \nsince raced',
    'cyclist last race \ndistance from race',
    # 'time trial \nstages count',
    # how should I call this feature? the indicative word (first word) is cyclist
    'workouts avg hr \nlast year avg deviation',
    'workouts avg calories \nlast year avg deviation',
    'workouts avg elevation gain \nlast year avg deviation',
    'workouts avg power \nlast year avg deviation',
    'workouts avg elevation loss \nlast year avg deviation',
    'workouts avg elevation \nlast year avg deviation',
    'workouts avg duration \nlast year avg deviation',
    'workouts avg temp \nlast year avg deviation',
    'workouts avg distance \nlast year avg deviation',
    'workouts avg TSS \nlast year avg deviation',
    'workouts avg IF \nlast year avg deviation',
    'workouts avg speed \nlast year avg deviation',
    'workouts avg cadence \nlast year avg deviation',
    'workouts avg hr \nlast year avg deviation',
    'workouts avg norm power \nlast year avg deviation',
    'workouts avg energy \nlast year avg deviation',
    'race difficulty level',  # e.g., Hard, Intermediate, Easy
    'race continent',
    'stage ranking',
    'stage profile score',
    'stage profile type',  # i.e., hilly, mountains,flat...
    'stage distance',
    'stage elevation gain',
    'stage temp avg',
    'race classification',  # i.e., WorldTour 1.WT
    'race total \nstages distance',
    'race total stages \nelevation gain',
    'race number \nof stages',
    'race has flat profile',
    'race has hills and flat finish',
    'race has hilly profile',
    'race has mountains and hilly finish',
    'race has mountains flat finish',
    'cyclist sprint points',
    'cyclist climber points',
    'cyclist one day \nraces points',
    'cyclist gc points',
    'cyclist time trial points'
]

tp_features_names_dict = {k: v for k, v in
                          zip(tp_final_features_before_dropping_thresholds, tp_final_features_before_dropping_thresholds_formal)}
strava_features_names_dict = {k: v for k, v in
                          zip(strava_final_features_before_dropping_thresholds, strava_final_features_before_dropping_thresholds_formal)}

BAR_PLOT_COLORS_DICT = {'race':'#1e90ff','stage':'#1e90ff','workouts':'#7eb54e','cyclist':'#f39C12'}

ONE_DAY_RACES,MAJOR_TOURS, GRAND_TOURS = "One Day Races","Major Tours","Grand Tours"

curve_factor_x = 'recalls'  # 'precision_recall_curve'
curve_factor_y = 'precisions'
EXEC_NAME = "Oct 30 - rewrite exprs"#"Sep 6-rerun - J paper results"#
EXEC_BASE_PATH = f"./executions/{EXEC_NAME}"#f"M:/Maor/Expr/maors_code/executions/{EXEC_NAME}"
WORKOUTS_SRC = 'STRAVA' #'STRAVA'
agg_func = 'SmartAgg'
POPULARITY_IN_CONTINENT = "Popularity in continent"
POPULARITY_IN_GENERAL = "Popularity in general"
POPULARITY_IN_RACE_CLASS = "Popularity in race class"  # classification
POPULARITY_IN_RACE_TYPE = "Popularity in race type"  # one day/stage race
baseline_algorithms = {"cyclist_popularity_ranking_in_team": POPULARITY_IN_GENERAL,
                       "cyclist_popularity_ranking_in_continent_in_team": POPULARITY_IN_CONTINENT,
                       "cyclist_popularity_ranking_in_race_class_in_team": POPULARITY_IN_RACE_CLASS,
                       "cyclist_popularity_ranking_in_race_type_in_team": POPULARITY_IN_RACE_TYPE,
                       }
features_names_dict = tp_features_names_dict if WORKOUTS_SRC == 'TP' else strava_features_names_dict
features_color_dict = {f: BAR_PLOT_COLORS_DICT[f.split()[0]] for f in features_names_dict.values()}

SINGLE_RACE_TYPE = None
SINGLE_TEAM = "Israel - Premier Tech" # "AG2R Citroën Team"  # "Israel - Premier Tech"
PLOT_AUC_PR = False

AUC_PR_INTERACTION = False


PLOT_EXPR_RESULTS = True
with_baseline = True

result_list =[] #[f"{EXEC_BASE_PATH}/['without', 5, 'STRAVA', 'Israel - Premier Tech']/[0.4, 'SimpleImputer']/['CatBoost']/ModelResults.csv"]  # [f"{EXEC_BASE_PATH}/[0.7, 0.7, 'SimpleImputer', 'without', 5, 'SmartAgg', 'TP', 'Israel - Premier Tech']/['CatBoost', None]/ModelResults.csv"]
baseline_results_path = f"{EXEC_BASE_PATH}/Final_Baselines_Results.csv"
model_results_path = f"{EXEC_BASE_PATH}/Final_Model_Results.csv"

PLOT_TIME = False
top_i = 8

PLOT_KN_RECALLS = False
xpoints = range(K_POINTS)

global_best_params_dict = {
    'col_threshold': '0.4',#None,#'0.5',
    'imputer': 'SimpleImputer',
    'imputer_workouts': 'without',
    # 'result_consideration':'[4,2,1]',
    'time_window': 5,
    'model': 'CatBoost'
}

team_best_params_dict={'AG2R Citroën Team': {'col_threshold': '0.45'},
                       'CCC Team': {'col_threshold': '0.45'},
                       'Team Jumbo-Visma':{'col_threshold': '0.45'},
                      'Lotto Fix All': {'col_threshold': '0.45'},
                       'Israel - Premier Tech':{'col_threshold': '0.45'},
                       'Cofidis': {'col_threshold': '0.45'},
                       'Groupama - FDJ': {'col_threshold': '0.45'},
                       'Movistar Team': {'col_threshold': '0.45'},
                       'UAE Team Emirates': {'col_threshold': '0.45'},
                       'Trek - Segafredo': {'col_threshold': '0.45'}
                       }


PLOT_ONLY_BEST = False

params_to_plot = {'imputer': 'Imputer Type',
                  # 'imputer_workouts': 'Workouts Imputer Type',
                  'model': 'Model',
                  # 'result_consideration':'Results Consideration',
                  # 'row_threshold': 'Row Threshold',
                  # 'col_threshold': 'Column Threshold',
                  # 'time_window': 'Time Window Size'
                  }
imputation_labels = {'without': 'Without imputation', 'SimpleImputer': 'With imputation - SimpleImputer',
                     'KNNImputer': 'With imputation - KNNImputer'}  # {'without': 'Without imputation', 'SimpleImputer': 'With imputation'}
model_labels = {'DecisionTree':'Decision Tree', 'CatBoost':'CatBoost', 'Logistic':'Logistic Regression'}
models_to_plot = ['DecisionTree', 'CatBoost', 'Logistic']
# BEST_RUN_MODEL_DIR = f"['{best_params_dict['model']}', None]"



PLOT_FEATURE_IMPORTANCE = False
FI_IN_TABLE = False
FI_RANKING_TABLE = False

SHAP, IG, RF, CATBST, CHI = True, True, True, True, True
RFF= False
# BEST_MODEL_ITER = f"4_{best_params_dict['model']}.pkl"
random_state = 0
number_of_races_to_test_fi = 5
interaction_feature = None#"distance_from_last_race"

PLOT_CATBOOST_TREE=False


PLOT_TREE = False
max_depth = 4
# tree_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/['DecisionTree', None]/0_DecisionTree.pkl"
# tree_img_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/['DecisionTree', None]"


# baseline_results_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/BaselinesResults.csv"
# model_results_path = f"{EXEC_BASE_PATH}/{BEST_RUN_DATA_DIR}/{BEST_RUN_MODEL_DIR}/ModelResults.csv"
