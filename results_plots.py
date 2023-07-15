import os

import catboost
import shap
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
import sklearn_relief as relief
import pickle
from sklearn import tree
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import auc

from expr_consts import SCORE_MODELS_COLS, MODELS_COLS
from results_consts import *
import matplotlib.pyplot as plt
import matplotlib
import json
from catboost import Pool

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
stages = pd.read_csv('./db/filtered_stages.csv')
races_number_of_days = {k: v['stage_id'] for k, v in
                        stages[['race_id', 'stage_id']].groupby('race_id').count().iterrows()}
races_days_ranges = {ONE_DAY_RACES: range(1, 2), MAJOR_TOURS: range(2, 13), GRAND_TOURS: range(13, 22)}
scores_cols_to_plot=['score_model_split']+SCORE_MODELS_COLS

def plot_auc_interaction_results(curve_factor_x, curve_factor_y, param_1, param_2, model_results):
    for team in model_results['team_name'].unique():
        if (SINGLE_TEAM is not None) and SINGLE_TEAM != team:
            continue
        sep_model_results_df = model_results.copy()
        sep_model_results_df = sep_model_results_df[sep_model_results_df['team_name'] == team]
        plt.style.use('ggplot')
        sep_model_results_df['auc_pri'] = sep_model_results_df[[curve_factor_x, curve_factor_y]].T.apply(
            lambda r: auc(r[0], r[1]))
        print(f'AUCPR - {params[param_1]},{params[param_2]} Parameters')
        matplotlib.rc('figure', figsize=(5, 5))
        sep_model_results_df = sep_model_results_df.sort_values(param_1)
        ax = sns.pointplot(x=param_1, y='auc_pri', hue=param_2,
                           data=sep_model_results_df[[param_1, param_2, 'auc_pri']], ci="sd", capsize=.2, join=False)

        # if 'model' == param:
        # plt.xticks(fontsize=9, rotation=-45)
        # plt.xlim(0, 1+EXTRA_LIM_GAP)
        # plt.ylim(0, 1+EXTRA_LIM_GAP)
        # plt.xticks(fontsize=9)
        ax.set_ylabel("AUC-PR")
        ax.set_xlabel(f"{params[param_1]}")

        plt.title(f'AUCPR - {params[param_1]}, {params[param_2]}')
        plt.legend(prop={"size": 14})

        save_file_path = f'results_plots/{EXEC_NAME}'
        create_dir_if_not_exist(save_file_path)
        save_file_path += f'/{WORKOUTS_SRC}'
        create_dir_if_not_exist(save_file_path)
        save_file_path += f'/{team}'
        create_dir_if_not_exist(save_file_path)
        if PLOT_ONLY_BEST:
            if PLOT_ONLY_BEST_BY_TEAM:
                path_to_save = f'{save_file_path}/{params[param_1]}, {params[param_2]} -  {titles[curve_factor_y]} - BEST BY TEAM.png'
            else:
                path_to_save = f'{save_file_path}/{params[param_1]}, {params[param_2]} -  {titles[curve_factor_y]} - BEST.png'
        else:
            path_to_save = f'{save_file_path}/{params[param_1]}, {params[param_2]} -  {titles[curve_factor_y]}.png'
        plt.savefig(path_to_save)
        # plt.gcf().subplots_adjust(left=0.15, bottom=0.13, top=0.97, right=0.96)
        plt.show()


def plot_auc_results(curve_factor_x, curve_factor_y, model_results, baseline_results=None,
                     baseline=POPULARITY_IN_GENERAL,
                     params_to_plot=params):
    plt.style.use('ggplot')
    model_results['auc_pri'] = model_results[[curve_factor_x, curve_factor_y]].T.apply(lambda r: auc(r[0], r[1]))
    baseline_results['auc_pri'] = baseline_results[[curve_factor_x, curve_factor_y]].T.apply(lambda r: auc(r[0], r[1]))
    for param in params_to_plot:
        print(f'AUCPR - {params[param]} Parameter')
        # if param != 'model':
        #     continue
        matplotlib.rc('figure', figsize=(5, 5))
        if param == 'model':
            baseline_results['model'] = baseline_results['model'].apply(lambda x: 'Popularity')
        else:
            baseline_results[param] = baseline_results[param].apply(lambda x: f'Popularity - {x}')
        df_to_plot = pd.concat([model_results[[param, 'auc_pri']], baseline_results[[param, 'auc_pri']]])
        ax = sns.pointplot(x=param, y='auc_pri', data=df_to_plot, ci="sd")

        # if 'model' == param:
        plt.xticks(fontsize=9, rotation=-45)
        # plt.xlim(0, 1+EXTRA_LIM_GAP)
        if ZOOM_IN:
            plt.ylim(ZOOM_IN['y'][0], ZOOM_IN['y'][1] + EXTRA_LIM_GAP)
        else:
            plt.ylim(0, 1 + EXTRA_LIM_GAP)
        # plt.xticks(fontsize=9)
        ax.set_ylabel("AUC-PR")
        plt.title(f'AUCPR - {params[param]}')
        plt.legend(prop={"size": 14, "loc": "lower right"})
        plt.savefig(f'results_plots/AUC - {params[param]}.png')
        plt.show()


def get_model_result_df(curve_factor_x, curve_factor_y, list_of_files_path=None, total_results_path=None):
    if list_of_files_path is not None:
        model_results_df = pd.DataFrame()
        for path in list_of_files_path:
            model_results_df = pd.concat([model_results_df, pd.read_csv(path)])

    elif total_results_path is not None:
        model_results_df = pd.read_csv(total_results_path)
    else:
        raise ValueError('Model results source is missing')
    model_results_df[curve_factor_y] = model_results_df[curve_factor_y].apply(json.loads)
    model_results_df[curve_factor_x] = model_results_df[curve_factor_x].apply(json.loads)
    return model_results_df


def get_baseline_result_df(curve_factor_x, curve_factor_y, list_of_files_path=None, total_results_path=None):
    if list_of_files_path is not None:
        baseline_results_df = pd.DataFrame()
        for path in list_of_files_path:
            baseline_results_df = pd.concat([baseline_results_df, pd.read_csv(path)])
    elif total_results_path is not None:
        baseline_results_df = pd.read_csv(total_results_path)
    else:
        raise ValueError('Baselines results source is missing')
    baseline_results_df[curve_factor_y] = baseline_results_df[curve_factor_y].apply(json.loads)
    baseline_results_df[curve_factor_x] = baseline_results_df[curve_factor_x].apply(json.loads)
    return baseline_results_df


def interp(x, y, window=3):
    x_new, y_new = [], []
    for i in range(0, len(x), window):
        if i + window >= len(x):
            x_new.append(x.iloc[-1])
            y_new.append(sum(y[i:]) / len(y[i:]))
        else:
            x_new.append(x.iloc[i + round(window / 2)])
            y_new.append(sum(y[i:i + window]) / len(y[i:i + window]))
    return x_new, y_new


def plot_time_results_at_i(curve_factor_y, top_i, model_results, baseline_results=None,
                           baseline=POPULARITY_IN_GENERAL,
                           params_to_plot=params):
    stages = pd.read_csv('./db/stages.csv')
    race_dates = {k: v for k, v in zip(stages['race_id'], stages['race_date'])}
    model_results['race_date'] = model_results['race_id'].map(race_dates)
    if baseline_results is not None:
        baseline_results['race_date'] = baseline_results['race_id'].map(race_dates)
    iter_to_date = {k: v for k, v in zip(model_results['iteration'], model_results['race_date'])}
    plt.style.use('ggplot')
    for w_src in workout_src_dict.keys():
        for param in params_to_plot:
            print(f'{params[param]} - {w_src}')
            # if param != 'model':
            #     continue
            matplotlib.rc('figure', figsize=(10, 10))
            fig, ax = plt.subplots()

            # Model graphs
            sep_model_results_df = model_results[model_results['workouts_source'] == workout_src_dict[w_src]]
            grouped_results = sep_model_results_df.groupby([param, 'iteration'])
            model_mean_results = pd.DataFrame(columns=[param, 'iteration', curve_factor_y])
            for n, g in grouped_results:
                if n[0] in ['KNN', 'RandomForest', 'GradientBoosting', 'LGBM', 'XGBoost', 'GaussianNB', 'Logistic',
                            'AdaBoost']:
                    continue
                top_i_factor = g[curve_factor_y].apply(
                    lambda x: x[(top_i - 1) + 1])  # +1 because we start at precision@0
                model_mean_results = model_mean_results.append(
                    [{param: n[0], 'iteration': n[1], curve_factor_y: top_i_factor.values[0]}])

            # print(model_mean_results)

            for n, g in model_mean_results.groupby(param):
                x, y = interp(g['iteration'].map(iter_to_date).iloc[::-1], g[curve_factor_y].iloc[::-1])
                ax.plot(x, y, label=n)
                print(f"Model Top {top_i}: {[round(e, 2) for e in g[curve_factor_y]]}")
                # print(f"Model Iteration: {g['iteration'].map(iter_to_date)}")
                # ax.plot(list(recall_new_points), precision_new_points, label=r[param])

            if baseline_results is not None:
                # Baseline graphs
                baseline_mean_results = pd.DataFrame(columns=[param, 'iteration', curve_factor_y])

                if param in ['model']:
                    grouped_results = baseline_results.groupby('iteration')
                    for iter, g in grouped_results:
                        top_i_factor = g[curve_factor_y].apply(
                            lambda x: x[(top_i - 1) + 1])  # +1 because we start at precision@0
                        mean_top_i = sum(top_i_factor.values) / len(top_i_factor)
                        baseline_mean_results = baseline_mean_results.append(
                            [{param: param, 'iteration': iter, curve_factor_y: mean_top_i}])

                    # ax.plot(list(recall_new_points), precision_new_points,
                    #         label=f'{baseline} baseline')
                    x, y = interp(baseline_mean_results['iteration'].map(iter_to_date).iloc[::-1],
                                  baseline_mean_results[curve_factor_y].iloc[::-1])
                    ax.plot(x, y, label=f'{baseline} baseline')
                    print(
                        f"Baseline Top {top_i}: {[round(e, 2) for e in baseline_mean_results[curve_factor_y].values]}")
                    # print(f"Baseline Iteration: {baseline_mean_results['iteration'].map(iter_to_date)}")
                else:
                    baseline_mean_results = pd.DataFrame(columns=[param, 'iteration', curve_factor_y])
                    grouped_results = baseline_results.groupby([param, 'iteration'])
                    for n, g in grouped_results:
                        top_i_factor = g[curve_factor_y].apply(
                            lambda x: x[(top_i - 1) + 1])  # +1 because we start at precision@0
                        baseline_mean_results = baseline_mean_results.append(
                            [{param: n[0], 'iteration': n[1], curve_factor_y: top_i_factor.values[0]}])

                    # print(baseline_mean_results)
                    for n, g in baseline_mean_results.groupby(param):
                        x, y = g['iteration'].map(iter_to_date).iloc[::-1], g[curve_factor_y].iloc[::-1]
                        ax.plot(x, y, label=n)
                        print(f"Baseline Top {top_i}: {[round(e, 2) for e in g[curve_factor_y].values]}")
                        # print(f"Baseline Iteration: {g['iteration'].map(iter_to_date)}")

            # if 'model' == param:
            plt.xticks(fontsize=9, rotation=-90)
            # plt.xlim(0, 1+EXTRA_LIM_GAP)
            # plt.ylim(0, 1+EXTRA_LIM_GAP)
            plt.xticks(fontsize=9)
            ax.set_ylabel(f"{titles[curve_factor_y][:-1]}{top_i}")
            ax.set_xlabel(f"Race/Time")
            plt.title(f'{params[param]} - {w_src}')
            plt.legend(prop={"size": 16})
            plt.savefig(f'results_plots/Time {titles[curve_factor_y][:-1]}{top_i} - {w_src}.png')
            plt.show()


def plot_pr_graphs(curve_factor_x, curve_factor_y, model_results, baseline_results=None,
                   params_to_plot=params_to_plot, x_points=None, cut_edges=False, x_label=None, team=None):
    if (team is not None) and (SINGLE_TEAM is not None) and SINGLE_TEAM != team:
        return
    elif WITH_MODEL_BASELINE:
        without_score_model_results = model_results[model_results['score_model'].isna()]
        if team is None:
            without_score_model_results, _ = get_all_teams_best_results_df(without_score_model_results, baseline_results)
        else:
            without_score_model_results, _ = get_best_results_df(without_score_model_results, baseline_results,team)
        without_score_model_results['baseline'] = RACEFIT_CYCLIST_STAGE
        baseline_results = pd.concat([baseline_results, without_score_model_results])

    if SINGLE_RACE_TYPE:
        model_relevant_races_pred = model_results['race_id'].map(races_number_of_days).isin(
            races_days_ranges[SINGLE_RACE_TYPE])
        baselines_relevant_races_pred = baseline_results['race_id'].map(races_number_of_days).isin(
            races_days_ranges[SINGLE_RACE_TYPE])
        model_results = model_results.loc[model_relevant_races_pred]
        baseline_results = baseline_results.loc[baselines_relevant_races_pred]
    if PLOT_ONLY_BEST:
        model_results_team, baseline_results_team = get_best_results_df(model_results, baseline_results,team,with_score_params=True)
        plot_graphs(baseline_results_team, curve_factor_x, curve_factor_y, cut_edges, model_results_team, team,
                    x_label, x_points, params_to_plot)
    else:
        for param in params_to_plot:
            if WITHOUT_SCORE_MODEL or (param in scores_cols_to_plot):
                plot_graphs(baseline_results, curve_factor_x, curve_factor_y, cut_edges, model_results, team,
                            x_label, x_points, params_to_plot, param=param)


def plot_pr_results(curve_factor_x, curve_factor_y, model_results, baseline_results=None,
                    params_to_plot=params_to_plot, x_points=None, cut_edges=False, x_label=None):
    plt.style.use('ggplot')
    if x_points is not None:
        model_results[curve_factor_y] = model_results[curve_factor_y].apply(lambda x: json.loads(x))
        if WITH_BASELINE:
            baseline_results[curve_factor_y] = baseline_results[curve_factor_y].apply(lambda x: json.loads(x))

    # for w_src in workout_src_dict.keys():
    if PLOT_ALL_TEAMS_AVG:
        plot_pr_graphs(curve_factor_x, curve_factor_y, model_results, baseline_results,
                       params_to_plot, x_points, cut_edges, x_label)
    else:
        for team in model_results['team_name'].unique():
            if team in TEAMS_TO_IGNORE:
                continue
            plot_pr_graphs(curve_factor_x, curve_factor_y, model_results, baseline_results,
                           params_to_plot, x_points, cut_edges, x_label, team)


def create_dir_if_not_exist(save_file_path):
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)


def plot_graphs(baseline_results, curve_factor_x, curve_factor_y, cut_edges, model_results, team, x_label,
                x_points, params, param=None):
    if team is not None:
        log_msg = f'{team} Precision - Recall {params[param] if param else "Best Parameters"}'
    else:
        log_msg = f'Precision - Recall {params[param] if param else "Best Parameters"}'
    print(log_msg)
    # if param != 'model':
    #     continue
    matplotlib.rc('figure', figsize=(15, 15))
    fig, ax = plt.subplots()
    # Model graphs
    # sep_model_results_df = model_results[model_results['workouts_source'] == workout_src_dict[w_src]]
    sep_model_results_df = model_results
    sep_model_results_to_group = sep_model_results_df
    if team is not None:
        sep_model_results_to_group = sep_model_results_df[sep_model_results_df['team_name'] == team]
    plot_graph(ax, curve_factor_x, curve_factor_y, cut_edges, param, sep_model_results_to_group, x_points)
    if baseline_results is not None:
        # Baselines graphs
        baseline_results_to_group = baseline_results
        if team is not None:
            baseline_results_to_group = baseline_results[baseline_results['team_name'] == team]
        grouped_results = baseline_results_to_group.groupby(['baseline'])
        for n, g in grouped_results:
            if (param not in scores_cols_to_plot+[None]) and n==RACEFIT_CYCLIST_STAGE:
                continue
            precision_points = (np.array(list(g[curve_factor_y])).sum(
                axis=0) / len(g))
            precision_points = precision_points[1:-10] if cut_edges else precision_points
            if x_points is None:
                recall_points = (np.array(list(g[curve_factor_x])).sum(
                    axis=0) / len(g))
            else:
                recall_points = x_points
            recall_points = recall_points[1:-10] if cut_edges else recall_points
            # ax.plot(list(recall_new_points), precision_new_points,
            #         label=f'{baseline} baseline')
            ax.plot(recall_points, precision_points,
                    label=f'{baseline_algorithms[n]} baseline', linewidth=LINE_WIDTH)
            print(f"Baseline Precision: {[round(e, 2) for e in precision_points]}")
            print(f"Baseline Recall: {[round(e, 2) for e in recall_points]}")
    # if 'model' == param:
    #     plt.xticks(fontsize=9, rotation=-45)
    if x_points is None:
        if ZOOM_IN:
            plt.xlim(ZOOM_IN['x'][0], ZOOM_IN['x'][1] + EXTRA_LIM_GAP)
        else:
            plt.xlim(0, 1 + EXTRA_LIM_GAP)
    else:
        # deprecated code
        x_points = x_points[1:-10] if cut_edges else x_points
        plt.xlim(x_points[0] - 0.0001, x_points[-2] + EXTRA_LIM_GAP)
    if ZOOM_IN:
        plt.ylim(ZOOM_IN['y'][0], ZOOM_IN['y'][1] + EXTRA_LIM_GAP)
    else:
        plt.ylim(0, 1 + EXTRA_LIM_GAP)
    plt.xticks(fontsize=42)
    plt.yticks(fontsize=42)
    ax.set_ylabel(titles[curve_factor_y], fontsize=44)
    if x_points is None:
        ax.set_xlabel(titles[curve_factor_x], fontsize=44)
    elif x_label is not None:
        ax.set_xlabel(x_label, fontsize=44)
    # plt.title(f'{params[param]} - {titles[curve_factor_y]}')
    plt.legend(prop={"size": 34})  # , loc="upper right")
    save_file_path = f'results_plots/{EXEC_NAME}'
    # pic_name=f"{params[param]} -  {titles[curve_factor_y]}"
    if param:
        pic_name = f"{param}_{titles_short[curve_factor_y]}"
    else:
        pic_name = f"_{titles_short[curve_factor_y]}"
        # pic_name = f"{sep_model_results_df.iloc[0][['time_window','col_threshold','imputer','model']].values}_{titles_short[curve_factor_y]}"
    create_dir_if_not_exist(save_file_path)
    file_name = ''
    # file_name = f'{WORKOUTS_SRC} '
    save_file_path += f'/{WORKOUTS_SRC}'
    create_dir_if_not_exist(save_file_path)
    if team is not None:
        save_file_path += f'/{team}'
        # file_name+=f'{team} '
        create_dir_if_not_exist(save_file_path)
    if WITHOUT_SCORE_MODEL:
        save_file_path += f'/WITHOUT_SCORE_MODEL'
        create_dir_if_not_exist(save_file_path)
    if SINGLE_RACE_TYPE:
        save_file_path += f'/{SINGLE_RACE_TYPE}'
        # file_name += f'{SINGLE_RACE_TYPE} '
        create_dir_if_not_exist(save_file_path)
    if PLOT_ONLY_BEST:
        if PLOT_ONLY_BEST:
            if PLOT_ONLY_BEST_BY_TEAM:
                file_name += f"{pic_name}-BEST BY TEAM.png"
            else:
                file_name += f"{pic_name}-BEST.png"
        save_file_path += f'/{file_name}'
    else:
        file_name += f"{pic_name}.png"
        save_file_path += f'/{file_name}'
    plt.savefig(save_file_path)
    plt.gcf().subplots_adjust(left=0.115, bottom=0.09, top=0.97, right=0.96)
    plt.show()


def plot_graph(ax, curve_factor_x, curve_factor_y, cut_edges, param, sep_model_results_to_group, x_points):

    if param is not None:
        grouped_results = sep_model_results_to_group.groupby(param)
    else:
        grouped_results = [("RaceFit",sep_model_results_to_group)]
    if x_points is None:
        model_mean_results = pd.DataFrame(columns=[param, curve_factor_y, curve_factor_x])
    else:
        model_mean_results = pd.DataFrame(columns=[param, curve_factor_y])
    for n, g in grouped_results:
        print(n)
        # if n in ['KNN', 'RandomForest', 'GradientBoosting', 'LGBM', 'XGBoost', 'GaussianNB', 'Logistic',
        #          'AdaBoost']:
        #     continue
        if x_points is None:
            model_mean_results = model_mean_results.append([{param: n, curve_factor_y: np.nansum(
                np.array(list(g[curve_factor_y])), axis=0) / len(g), curve_factor_x: np.nansum(
                np.array(list(g[curve_factor_x])), axis=0) / len(g)}])
        else:
            model_mean_results = model_mean_results.append([{param: n, curve_factor_y: np.nansum(
                np.array(list(g[curve_factor_y])), axis=0) / len(g)}])
    # print(model_mean_results)
    colors_imputation = ['#74b362', '#e0991d']
    # colors_classifier=['#e377c2','#74b362','#7f7f7f']
    j = 0
    for i, r in model_mean_results.iterrows():
        if x_points is None:
            recall_points = list(r[curve_factor_x])[1:-10] if cut_edges else list(r[curve_factor_x])
        else:
            recall_points = x_points[1:-10] if cut_edges else x_points
        precision_points = list(r[curve_factor_y])[1:-10] if cut_edges else list(r[curve_factor_y])
        label = r[param]
        if param == 'imputer':
            label = f"{imputation_labels[label]}"
            ax.plot(recall_points, precision_points, label=label, linewidth=LINE_WIDTH, color=colors_imputation[j])
        elif param in ['model', 'score_model']:
            label = f"{model_labels[label]}"
            ax.plot(recall_points, precision_points, label=label, linewidth=LINE_WIDTH)
        elif param == 'time_window':
            label = f"{label} weeks"
            ax.plot(recall_points, precision_points, label=label, linewidth=LINE_WIDTH)
        else:
            ax.plot(recall_points, precision_points, label=label, linewidth=LINE_WIDTH)
        print(f"Model Precision: {[round(e, 2) for e in precision_points]}")
        print(f"Model Recall: {[round(e, 2) for e in recall_points]}")
        # ax.plot(list(recall_new_points), precision_new_points, label=r[param])
        j += 1


def plot_bar_feature_selection(team, names, importances, method):
    fig, ax = plt.subplots(figsize=(22, 10))
    importances = [round(e, 2) for e in importances]
    features_colors = [features_color_dict[features_names_dict[n]] for n in names]
    names = [features_names_dict[n].replace('\n', '') for n in names]  # [features_names_dict[n] for n in names]#
    bar = ax.barh(names, importances, color=features_colors)
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    ax.invert_yaxis()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # ax.bar_label(bar,fontsize=22)
    # ax.set_title(f'Feature importance - {method}')
    legend_dict = {'Race': '#1e90ff', 'Workout': '#7eb54e', 'Cyclist': '#f39C12'}
    labels = list(legend_dict.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=legend_dict[label]) for label in labels]
    plt.legend(handles, labels, prop={'size': 22}, bbox_to_anchor=(-1.1, 1.01))
    plt.gcf().subplots_adjust(left=0.6, right=0.98, top=0.98, bottom=0.06)
    plt.savefig(f'results_plots/{EXEC_NAME}/{WORKOUTS_SRC}/{team}/Feature importance - {method}.png')
    plt.show()


def load_model(model_path):
    import pickle
    return pickle.load(open(model_path, 'rb'))


def get_shap_values(model, X_test):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_test)


def get_correlated_features(X):
    correlation_matrix = X.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                col = correlation_matrix.columns[i]
                correlated_features.add(col)
    return correlated_features


def drop_correlated_features(X, columns_to_ignore):
    X_to_ignore = X[columns_to_ignore]
    X = X.drop(columns=columns_to_ignore)
    correlated_features = get_correlated_features(X)
    print(correlated_features)
    X = X.drop(columns=correlated_features)
    X[columns_to_ignore] = X_to_ignore
    return X


def get_test_train_by_race(X_path, number_of_races_to_test_fi):
    import pandas as pd
    X = pd.read_csv(X_path)
    race_ids = X['race_id'].unique()[:number_of_races_to_test_fi]
    X_test = X[X['race_id'].isin(race_ids)]
    X_train = X[~X['race_id'].isin(race_ids)]
    return X_train, X_test


def get_information_gain_scores(X, Y, threshold=10):
    high_score_features, scores = [], []
    feature_scores = mutual_info_classif(X, Y, random_state=0)
    zipped = [x for x in zip(feature_scores, X.columns) if str(x[0]) != 'nan']
    sorted_list = sorted(zipped, reverse=True)
    if threshold is None:
        iterable = sorted_list
    else:
        iterable = sorted_list[:threshold]
    for score, f_name in iterable:
        # print(f_name, score)
        high_score_features.append(f_name)
        scores.append(score)
    return high_score_features, scores


def plot_information_gain(team, X_train, Y_train):
    high_score_features, scores = get_information_gain_scores(X_train, Y_train)
    plot_bar_feature_selection(team, high_score_features, scores, "Information Gain")


def get_relieff_scores(X, Y, threshold=10):
    high_score_features, scores = [], []
    if threshold is None:
        threshold = len(X.columns)
    fs = relief.ReliefF(n_features=threshold, random_state=random_state)
    fs.fit_transform(np.array(X), np.array(Y).reshape(-1))
    features_w = {}
    for i in range(len(fs.w_)):
        col = X.columns[i]
        features_w[col] = fs.w_[i]
    for f_name, score in sorted(features_w.items(), key=lambda item: item[1], reverse=True)[:threshold]:
        print(f_name, score)
        high_score_features.append(f_name)
        scores.append(score)
    return high_score_features, scores


def get_relief_scores(X, Y, threshold=10):
    high_score_features, scores = [], []
    if threshold is None:
        threshold = len(X.columns)
    fs = relief.Relief(n_features=threshold, random_state=random_state)
    fs.fit_transform(np.array(X), np.array(Y).reshape(-1))
    features_w = {}
    for i in range(len(fs.w_)):
        col = X.columns[i]
        features_w[col] = fs.w_[i]
    for f_name, score in sorted(features_w.items(), key=lambda item: item[1], reverse=True)[:threshold]:
        print(f_name, score)
        high_score_features.append(f_name)
        scores.append(score)
    return high_score_features, scores


def plot_chi(team, X_train, Y_train):
    high_score_features, scores = get_chi_scores(X_train, Y_train)
    plot_bar_feature_selection(team, high_score_features, scores, "Chi Square")


def plot_relief(team, X_train, Y_train):
    high_score_features, scores = get_relief_scores(X_train, Y_train)
    plot_bar_feature_selection(team, high_score_features, scores, "Relief")


def get_catboost_scores(X_train, model, threshold=10):
    high_score_features, scores = [], []
    features_names = list(X_train.columns)
    fi_scores = model.get_feature_importance()
    zipped = [x for x in zip(fi_scores, features_names) if str(x[0]) != 'nan']
    sorted_list = sorted(zipped, reverse=True)
    if threshold is None:
        iterable = sorted_list
    else:
        iterable = sorted_list[:threshold]
    for score, f_name in iterable:
        # print(f_name, score)
        high_score_features.append(f_name)
        scores.append(score)
    return high_score_features, scores


def plot_catboost(team, X_train, model):
    high_score_features, scores = get_catboost_scores(X_train, model)
    plot_bar_feature_selection(team, high_score_features, scores, "CatBoost")


def get_best_run_data_dict(team):
    result_dict = global_best_params_dict.copy()
    for k in result_dict.keys():
        if PLOT_ONLY_BEST_BY_TEAM and (k in team_best_params_dict[team]):
            result_dict[k] = team_best_params_dict[team][k]
    return result_dict


def get_chi_scores(X_train, Y_train, threshold=10):
    X_min_value = X_train.min()
    for f in X_min_value.index:
        if X_min_value[f] < 0:
            X_train[f] = X_train[f] + abs(X_min_value[f])
    X_new = SelectKBest(chi2, k='all')
    X_new.fit_transform(X_train, Y_train)
    high_score_features, scores = [], []
    zipped = [x for x in zip(X_new.scores_, X_new.feature_names_in_) if str(x[0])!='nan']
    sorted_list = sorted(zipped, reverse=True)
    if threshold is None:
        iterable = sorted_list
    else:
        iterable = sorted_list[:threshold]
    for score, f_name in iterable:
        # print(f_name, score)
        high_score_features.append(f_name)
        scores.append(score)
    return high_score_features, scores


def create_feature_importance_tables(drop_corr=False, score_table=True, ranking_table=False):
    ig_df, rf_df, rff_df, shap_df, cat_df, chi_df = [None] * 6
    for team in TEAM_NAMES.values():
        if (SINGLE_TEAM is not None) and SINGLE_TEAM != team:
            continue
        if team in TEAMS_TO_IGNORE:
            continue
        best_run_data_dir, _ = get_best_run_dir(team)
        X_path = f"{EXEC_BASE_PATH}/{best_run_data_dir}/X_cols_data.csv"
        Y_path = f"{EXEC_BASE_PATH}/{best_run_data_dir}/Y_cols_data.csv"
        X_train, X_test = get_test_train_by_race(X_path, number_of_races_to_test_fi)
        Y = pd.read_csv(Y_path)
        Y_train, Y_test = Y.loc[X_train.index], Y.loc[X_test.index]
        if IG:
            if drop_corr:
                X_train = drop_correlated_features(X_train, ['stage_id', 'cyclist_id', 'race_id'])
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            high_score_features, scores = get_information_gain_scores(X_train_filtered, Y_train, None)
            high_score_features = [features_names_dict[n] for n in high_score_features]
            scores_dict = dict(zip(high_score_features, scores))
            if ig_df is None:
                ig_df = pd.DataFrame(columns=scores_dict.keys())
            ig_df.loc[team] = pd.Series(scores_dict)
        if RF:
            if drop_corr:
                X_train = drop_correlated_features(X_train, ['stage_id', 'cyclist_id', 'race_id'])
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            high_score_features, scores = get_relief_scores(X_train_filtered, Y_train, None)
            high_score_features = [features_names_dict[n] for n in high_score_features]
            scores_dict = dict(zip(high_score_features, scores))
            if rf_df is None:
                rf_df = pd.DataFrame(columns=scores_dict.keys())
            rf_df.loc[team] = pd.Series(scores_dict)
        if RFF:
            if drop_corr:
                X_train = drop_correlated_features(X_train, ['stage_id', 'cyclist_id', 'race_id'])
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            high_score_features, scores = get_relieff_scores(X_train_filtered, Y_train, None)
            high_score_features = [features_names_dict[n] for n in high_score_features]
            scores_dict = dict(zip(high_score_features, scores))
            if rff_df is None:
                rff_df = pd.DataFrame(columns=scores_dict.keys())
            rff_df.loc[team] = pd.Series(scores_dict)
        if SHAP:
            model = load_last_model(team, index=4)
            X_test = X_test.drop(columns=['stage_id', 'cyclist_id', 'race_id'])
            shap_values = get_shap_values(model, X_test)
            vals = np.abs(shap_values).mean(0)
            high_score_features = [features_names_dict[n] for n in X_test.columns]
            scores_dict = dict(zip(high_score_features, vals))
            if shap_df is None:
                shap_df = pd.DataFrame(columns=scores_dict.keys())
            shap_df.loc[team] = pd.Series(scores_dict)
        if CATBST:
            model = load_last_model(team)
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            high_score_features, scores = get_catboost_scores(X_train_filtered, model, threshold=None)
            high_score_features = [features_names_dict[n] for n in high_score_features]
            scores_dict = dict(zip(high_score_features, scores))
            if cat_df is None:
                cat_df = pd.DataFrame(columns=scores_dict.keys())
            cat_df.loc[team] = pd.Series(scores_dict)
        if CHI:
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            high_score_features, scores = get_chi_scores(X_train_filtered, Y_train, threshold=None)
            high_score_features = [features_names_dict[n] for n in high_score_features]
            scores_dict = dict(zip(high_score_features, scores))
            if chi_df is None:
                chi_df = pd.DataFrame(columns=scores_dict.keys())
            chi_df.loc[team] = pd.Series(scores_dict)

    if IG:
        write_fi_to_csv(ig_df, "information_gain", score_table, ranking_table)
    if RF:
        write_fi_to_csv(rf_df, "relief", score_table, ranking_table)
    if RFF:
        write_fi_to_csv(rff_df, "relieff", score_table, ranking_table)
    if CATBST:
        write_fi_to_csv(cat_df, "catboost", score_table, ranking_table)
    if CHI:
        write_fi_to_csv(chi_df, "chi", score_table, ranking_table)
    if SHAP:
        write_fi_to_csv(shap_df, "shap", score_table, ranking_table)


def load_last_model(team, index=1):
    best_run_data_dir, best_run_data_dict = get_best_run_dir(team, True)
    result_consideration= f"'{best_run_data_dict['result_consideration']}'" if ('result_consideration' in best_run_data_dict and best_run_data_dict['result_consideration']) else 'None'
    score_model_split= f"'{best_run_data_dict['score_model_split']}'" if ('score_model_split' in best_run_data_dict and best_run_data_dict['score_model_split']) else 'None'
    catboost_dir = f"{EXEC_BASE_PATH}/{best_run_data_dir}/['{best_run_data_dict['model']}', {result_consideration}, {score_model_split}]"
    last_catboost_model_path = f"{catboost_dir}/{index}_{best_run_data_dict['model']}.pkl"
    model = pickle.load(open(last_catboost_model_path, 'rb'))
    return model


def get_best_run_dir(team, return_dict=False):
    best_run_data_dict = get_best_run_data_dict(team)
    best_run_data_dict['col_threshold'] = 1-(int(best_run_data_dict['col_threshold'][:-1])/100)
    best_run_data_dir = f"['without', {best_run_data_dict['time_window']}, '{WORKOUTS_SRC}', '{team}']/[{best_run_data_dict['col_threshold']}, '{best_run_data_dict['imputer']}']"
    return (best_run_data_dir, best_run_data_dict) if return_dict else (best_run_data_dir, None)


def write_fi_to_csv(df, method, score_table, ranking_table):
    create_dir_if_not_exist(f'results_plots/{EXEC_NAME}')
    create_dir_if_not_exist(f'results_plots/{EXEC_NAME}/{WORKOUTS_SRC}')
    df = df.apply(lambda f: round(f, 3))
    if ranking_table:
        ranking_df_helper = df.copy().T
        ranking_df = pd.DataFrame()
        for c in TEAMS_RANKING_DICT:
            if (SINGLE_TEAM is not None) and SINGLE_TEAM != c:
                continue
            if c in TEAMS_TO_IGNORE:
                continue
            new_column_values = ranking_df_helper[c].sort_values(ascending=False).apply(str)
            new_column_values.index = [n.replace('\n', '') for n in new_column_values.index]
            new_column_values = pd.Series(new_column_values.index + ' - ' + new_column_values.values).reset_index(
                drop=True)
            new_column_values.index = new_column_values.index + 1
            ranking_df[f"{TEAMS_RANKING_DICT[c]} - {c}"] = new_column_values.values
        ranking_df.to_csv(f'results_plots/{EXEC_NAME}/{WORKOUTS_SRC}/{method}_scores_ranking_table.csv', header=True,
                          index=False)
    if score_table:
        df['team_name'] = df.index
        df['ranking'] = df['team_name'].map(TEAMS_RANKING_DICT)
        df = df.sort_values('ranking')
        df = df.drop(columns=['ranking'])
        df = df.T
        df['feature'] = df.index
        df['feature'] = df['feature'].apply(lambda x: x.replace('\n', ''))
        df = df.reset_index(drop=True)
        df.to_csv(f'results_plots/{EXEC_NAME}/{WORKOUTS_SRC}/{method}_scores_table.csv', header=True, index=False)


def plot_feature_importance(number_of_races_to_test_fi, interaction_feature=None, drop_corr=False):
    for team in TEAM_NAMES.values():
        if (SINGLE_TEAM is not None) and SINGLE_TEAM != team:
            continue
        if team in TEAMS_TO_IGNORE:
            continue
        best_run_data_dir, _ = get_best_run_dir(team)
        X_path = f"{EXEC_BASE_PATH}/{best_run_data_dir}/X_cols_data.csv"
        Y_path = f"{EXEC_BASE_PATH}/{best_run_data_dir}/Y_cols_data.csv"
        X_train, X_test = get_test_train_by_race(X_path, number_of_races_to_test_fi)
        Y = pd.read_csv(Y_path)
        Y_train, Y_test = Y.loc[X_train.index], Y.loc[X_test.index]
        plt.style.use('default')
        if IG:
            if drop_corr:
                X_train = drop_correlated_features(X_train, ['stage_id', 'cyclist_id', 'race_id'])
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            plot_information_gain(team, X_train_filtered, Y_train)
        if RF:
            if drop_corr:
                X_train = drop_correlated_features(X_train, ['stage_id', 'cyclist_id', 'race_id'])
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            plot_relief(team, X_train_filtered, Y_train)
        if SHAP:
            model = load_last_model(team, index=5)
            X_test = X_test.drop(columns=['stage_id', 'cyclist_id', 'race_id'])
            plot_shap(team, X_test, interaction_feature, model)
        if CATBST:
            model = load_last_model(team)
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            plot_catboost(team, X_train_filtered, model)
        if CHI:
            if drop_corr:
                X_train = drop_correlated_features(X_train, ['stage_id', 'cyclist_id', 'race_id'])
            X_train_filtered = X_train.drop(columns=['cyclist_id', 'stage_id', 'race_id'])
            plot_chi(team, X_train_filtered, Y_train)


def plot_shap(team, X_test, interaction_feature, model, threshold=10):
    plt.clf()
    plt.gcf().subplots_adjust(left=0.35)
    features_names = list(X_test.columns)
    shap_values = get_shap_values(model, X_test)
    shap.summary_plot(shap_values, X_test, plot_size=(14, 10), show=False)  # plot_type='bar', max_display=X.shape[1]
    plt.savefig(f'results_plots/{EXEC_NAME}/{WORKOUTS_SRC}/{team}/Shap summary.png')
    high_score_features, scores = [], []
    fi_scores = np.abs(shap_values).mean(0)
    zipped = [x for x in zip(fi_scores, features_names) if str(x[0]) != 'nan']
    sorted_list = sorted(zipped, reverse=True)
    if threshold is None:
        iterable = sorted_list
    else:
        iterable = sorted_list[:threshold]
    for score, f_name in iterable:
        # print(f_name, score)
        high_score_features.append(f_name)
        scores.append(score)
    plot_bar_feature_selection(team, high_score_features, scores, "Shap")
    if interaction_feature:
        shap.dependence_plot(interaction_feature, shap_values, X_test, show=False)
        plt.savefig(f'results_plots/{EXEC_NAME}/{WORKOUTS_SRC}/{team}/Shap dependence plot.png')


def plot_tree(tree_path, X_path, Y_path, tree_img_path):
    tree_model = pickle.load(open(tree_path, 'rb'))
    X, Y = pd.read_csv(X_path), pd.read_csv(Y_path)
    fig = plt.figure(figsize=(190, 80))
    plt.gcf().subplots_adjust(left=-0.001, bottom=-0.0012, top=0.99, right=0.99)
    features_names = [features_names_dict[c] for c in X.drop(columns=['stage_id', 'cyclist_id', 'race_id']).columns]
    _ = tree.plot_tree(tree_model, max_depth=max_depth,
                       feature_names=features_names, rounded=True,
                       class_names=['0', '1'],
                       filled=True, fontsize=55)

    # text_representation = tree.export_text(tree_model,max_depth=max_depth,
    #                    feature_names=features_names)
    # print(text_representation)
    plt.savefig(f'{tree_img_path}/decision_tree.jpg')


def plot_catboost_tree():
    for team in TEAM_NAMES.values():
        _, best_run_data_dir = get_best_run_dir(team, True)
        model = load_last_model(team)

        Y = pd.read_csv(f"{EXEC_BASE_PATH}/{best_run_data_dir}/Y_cols_data.csv")
        X = pd.read_csv(f"{EXEC_BASE_PATH}/{best_run_data_dir}/X_cols_data.csv").drop(
            columns=['cyclist_id', 'stage_id', 'race_id'])
        # feature_names = [features_names_dict[n] for n in list(X.columns)]
        pool = Pool(X, Y, feature_names=list(X.columns))
        g = model.plot_tree(0, pool)

        save_file_path = f'results_plots/{EXEC_NAME}'
        create_dir_if_not_exist(save_file_path)
        save_file_path += f'/{WORKOUTS_SRC}'
        create_dir_if_not_exist(save_file_path)
        save_file_path += f'/{team}'
        create_dir_if_not_exist(save_file_path)
        g.format = 'png'
        g.render(filename=f'{save_file_path}/Catboost Tree - {team}.gv.png')

def get_best_results_df(model_results, baseline_results,team,with_score_params=False):
    if (SINGLE_TEAM is not None) and SINGLE_TEAM != team:
        return
    if team is None:
        return get_all_teams_best_results_df(model_results, baseline_results)
    best_run_data_dict = get_best_run_data_dict(team)
    for k in best_run_data_dict:
        if not with_score_params:
            if (not WITHOUT_SCORE_MODEL) and (k in scores_cols_to_plot):
                continue
        transform = lambda df, k: df[df[k] == best_run_data_dict[k]]
        if k == 'col_threshold':
            transform = lambda df, k: df[df[k].apply(str) == best_run_data_dict[k]]
        if best_run_data_dict[k] is None:
            transform = lambda df, k: df[df[k].isna()]
        model_results = transform(model_results, k)
        if WITH_BASELINE:
            if k in MODELS_COLS + scores_cols_to_plot:
                continue
            baseline_results = transform(baseline_results, k)
    return model_results, baseline_results

def get_all_teams_best_results_df(model_results, baseline_results):
    model_results_result,baseline_results_result = pd.DataFrame(), pd.DataFrame()
    for team in model_results['team_name'].unique():
        model_results_team = model_results[model_results['team_name'] == team]
        baseline_results_team = baseline_results[baseline_results['team_name'] == team]
        model_results_team, baseline_results_team = get_best_results_df(model_results_team, baseline_results_team,team)
        model_results_result = pd.concat([model_results_result,model_results_team])
        baseline_results_result = pd.concat([baseline_results_result, baseline_results_team])
    return model_results_result, baseline_results_result


if __name__ == '__main__':
    file_paths = []

    if PLOT_EXPR_RESULTS or PLOT_TIME or PLOT_KN_RECALLS or PLOT_AUC_PR or AUC_PR_INTERACTION:
        if result_list:
            if 'model' in result_list:
                for model in RESULTS_LISTS['model']:
                    file_paths.append(
                        f"{EXEC_BASE_PATH}/[0.85, 0.85, 'IterativeImputer', 'MinMaxScaler', 30, 'Average', 'STRAVA']/['{model}']/ModelResults.csv")
            if 'col_threshold' in result_list:
                # for col_t in RESULTS_LISTS['col_threshold']:
                #     file_paths.append(
                #         f"./executions/June 9 23.00 - change row threshold to be first/[0.7, {col_t}, 'KNNImputer', 'StandardScaler', 5, 'SmartAgg', 'STRAVA']/['CatBoost', None]/ModelResults.csv")
                file_paths.append(
                    f"{EXEC_BASE_PATH}/[0.7, 0.2, 'IterativeImputer', 'without', 5, 'SmartAgg', 'TP']/['CatBoost', None]/ModelResults.csv")
                file_paths.append(
                    f"{EXEC_BASE_PATH}/[0.7, 0.7, 'KNNImputer', 'StandardScaler', 5, 'SmartAgg', 'STRAVA']/['CatBoost', None]/ModelResults.csv")
            if 'team_name' in result_list:
                for team_id in RESULTS_LISTS['team_name']:
                    file_paths.append(
                        f"{EXEC_BASE_PATH}/[0.7, 0.7, 'SimpleImputer', 'without', 5, 'SmartAgg', 'STRAVA', '{TEAM_NAMES[team_id]}']/['CatBoost', None]/ModelResults.csv")
            else:
                file_paths = result_list.copy()
            model_results = get_model_result_df(curve_factor_x, curve_factor_y, list_of_files_path=file_paths)
        elif MULTIPLE_FILES:
            for res_file in os.listdir(f"{EXEC_BASE_PATH}/final"):
                if '_Model_' in res_file:
                    file_paths.append(
                        f"{EXEC_BASE_PATH}/final/{res_file}")
            model_results = get_model_result_df(curve_factor_x, curve_factor_y, list_of_files_path=file_paths)
        else:
            model_results = get_model_result_df(curve_factor_x, curve_factor_y, total_results_path=model_results_path)

        file_paths = []
        baseline_results = None
        if WITH_BASELINE:
            if MULTIPLE_FILES:
                for res_file in os.listdir(f"{EXEC_BASE_PATH}/final"):
                    if '_Baselines_' in res_file:
                        file_paths.append(
                            f"{EXEC_BASE_PATH}/final/{res_file}")
                baseline_results = get_baseline_result_df(curve_factor_x, curve_factor_y, list_of_files_path=file_paths)
            else:
                baseline_results = get_baseline_result_df(curve_factor_x, curve_factor_y,
                                                          total_results_path=baseline_results_path)
            baseline_results = baseline_results[baseline_results['iteration'].isin(model_results['iteration'])]

        model_results = model_results[~model_results['team_name'].isin(TEAMS_TO_IGNORE)]
        baseline_results = baseline_results[~baseline_results['team_name'].isin(TEAMS_TO_IGNORE)]
        model_results = model_results[model_results['col_threshold']!=0.8]
        baseline_results = baseline_results[baseline_results['col_threshold']!=0.8]
        model_results['score_model'] = model_results['score_model'].apply(lambda x: None if str(x)=='without' else x)
        baseline_results['score_model'] = baseline_results['score_model'].apply(lambda x: None if str(x)=='without' else x)

        # model_results['weighted_mean'] = model_results['weighted_mean'].apply(lambda x: 'without' if str(x)=='nan' else x)

        if WITHOUT_SCORE_MODEL:
            model_results = model_results[model_results['score_model'].isna()]
            baseline_results = baseline_results[baseline_results['score_model'].isna()]
            # model_results = model_results[model_results['score_model_split'] == 'Without']
            # baseline_results = baseline_results[baseline_results['score_model_split'] == 'Without']


        model_results = model_results[model_results['workouts_source'] == WORKOUTS_SRC]
        baseline_results = baseline_results[baseline_results['workouts_source'] == WORKOUTS_SRC]
        model_results['col_threshold'] = model_results['col_threshold'].apply(lambda x: str(round((1 - x) * 100)) + "%")
        baseline_results['col_threshold'] = baseline_results['col_threshold'].apply(lambda x: str(round((1 - x) * 100)) + "%")

        model_results['score_model_split'] = model_results['score_model_split'].apply(lambda x: str(round((1 - x) * 100)) + "%" if str(x)!='nan' else x)
        baseline_results['score_model_split'] = baseline_results['score_model_split'].apply(lambda x: str(round((1 - x) * 100)) + "%" if str(x)!='nan' else x)
        model_results['k_clusters'] = model_results['k_clusters'].apply(lambda x: "C "+str(int(x)) if str(x)!='nan' else x)
        baseline_results['k_clusters'] = baseline_results['k_clusters'].apply(lambda x: "C "+str(int(x)) if str(x)!='nan' else x)


        model_results = model_results[model_results['score_model_split'].isna()]



        # without 2022data
        # baseline_results=baseline_results[baseline_results['iteration']>=38]
        # model_results=model_results[model_results['iteration']>=38]

        # model_results = model_results[model_results['time_window'] == 5]
        # model_results['col_threshold'] = model_results['col_threshold'].apply(str)
        # model_results = model_results[model_results['col_threshold'] == '0.4']
        # model_results = model_results[model_results['imputer_workouts'] == 'without']

        # if WITH_BASELINE:
        # baseline_results = baseline_results[baseline_results['time_window'] == 5]
        # baseline_results['col_threshold'] = baseline_results['col_threshold'].apply(str)
        # baseline_results = baseline_results[baseline_results['col_threshold'] == '0.4']
        # baseline_results = baseline_results[baseline_results['imputer_workouts'] == 'without']






    if PLOT_TIME:
        plot_time_results_at_i(curve_factor_y, top_i, model_results, baseline_results=baseline_results,
                               baseline=POPULARITY_IN_GENERAL,
                               params_to_plot={'model': params['model']})

    if PLOT_EXPR_RESULTS:
        model_results = model_results[model_results['model'].isin(models_to_plot)]
        plot_pr_results(curve_factor_x, curve_factor_y, model_results, baseline_results=baseline_results,
                        cut_edges=True, params_to_plot=params_to_plot)

    if PLOT_KN_RECALLS:
        model_results = model_results[model_results['model'].isin(models_to_plot)]
        curve_factor_y = 'recalls_kn'
        plot_pr_results(None, curve_factor_y, model_results, baseline_results=baseline_results,
                        x_points=xpoints, x_label='k',
                        params_to_plot={'model': params['model']})

    if PLOT_AUC_PR:
        plot_auc_results(curve_factor_x, curve_factor_y, model_results, baseline_results=baseline_results)

    if AUC_PR_INTERACTION:
        params_interaction = [('time_window', 'col_threshold'),('col_threshold', 'imputer'),('imputer', 'model')]
        for param_1, param_2 in params_interaction:
            plot_auc_interaction_results(curve_factor_x, curve_factor_y, param_1, param_2, model_results)

    if PLOT_FEATURE_IMPORTANCE:
        if FI_IN_TABLE or FI_RANKING_TABLE:
            create_feature_importance_tables(score_table=FI_IN_TABLE, ranking_table=FI_RANKING_TABLE)
        else:
            plot_feature_importance(number_of_races_to_test_fi,
                                    interaction_feature=interaction_feature)  # ,drop_corr=True)

    # if PLOT_TREE:
    #     plot_tree(tree_path, X_path, Y_path, tree_img_path)

    if PLOT_CATBOOST_TREE:
        plot_catboost_tree()
