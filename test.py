e = []
t = pd.read_csv('./allocation_matrices/races_cyclist_matrix_Trek - Segafredo.csv')
for c in races_cyclist_matrix.columns:
    e.append(all((t[str(c)]== races_cyclist_matrix[c]).values))
assert all(e)

executions/Oct 3-specialty points added/['SimpleImputer', 5, 'STRAVA', 'Israel - Premier Tech']/X_cols_raw_data.csv
executions/Oct 3-specialty points added/['SimpleImputer', 5, 'STRAVA', 'Israel - Premier Tech']/Y_cols_raw_data.csv

comparison_list = []
true_df = pd.read_csv("executions/Oct 3-specialty points added/['SimpleImputer', 5, 'STRAVA', 'Israel - Premier Tech']/X_cols_raw_data.csv")
df_to_test = X
for c in true_df.columns:
    comparison_list.append(all((true_df[str(c)]== df_to_test[c]).values))
print(all(comparison_list))

comparison_list = []
true_df = pd.read_csv(
    "executions/Oct 3-specialty points added/['SimpleImputer', 5, 'STRAVA', 'Israel - Premier Tech']/X_cols_raw_data.csv")
df_to_test = X
for c in true_df.columns:
    c_x = c.replace('bias', 'deviation')

    true_df[str(c)] = true_df[str(c)].apply(lambda n: round(n, 2) if isinstance(n, str) and check_float(n) else n)
    df_to_test[c_x] = df_to_test[c_x].apply(lambda n: round(n, 2) if isinstance(n, str) and check_float(n) else n)

    test_c = all((true_df[str(c)][:15] == df_to_test[c_x][:15]).values)
    if not test_c:
        print(test_c, f' {c}')
    comparison_list.append(test_c)
print(all(comparison_list))