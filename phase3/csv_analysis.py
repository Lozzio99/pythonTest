import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

UB = pd.read_csv('../csv_competition/upperbound.csv')
instances_names = UB.instance
split_names = pd.read_csv('../split_csvs/SPLIT_BASELINE-SORTED_DECREASING_ITEM_HEIGHT-BOTTOM_LEFT.csv').instance


def load_fixed_ub_results():
    all_results = {}

    def load_results(fn, df):
        fn = fn.replace(".csv", "")
        fn = fn.replace("BASELINE_SORTED_", "")
        fn = fn.replace("ITEM_", "")
        fn = fn.replace("_STRIP_PACKER_FFDH", "")
        fn = fn.replace("DECREASING_", '-')
        fn = fn.replace("INCREASING_", '+')
        all_results[fn.lower()] = df

    def fix_upper_bound(fn):
        df = pd.read_csv(f'../csv_competition/{fn}')
        names = df['instance']
        for name in names:
            value = UB.loc[UB['instance'] == name, 'upper bound'].values[0]
            df.loc[df['instance'] == name, 'upper bound'] = value

        # df.to_csv(f'../fixed/{fn}')
        load_results(fn, df)
        return df

    fix_upper_bound('BASELINE_SORTED_DECREASING_ITEM_HEIGHT_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_DECREASING_ITEM_WIDTH_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_DECREASING_ITEM_VALUE_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_DECREASING_ITEM_RATIO_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_INCREASING_ITEM_HEIGHT_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_INCREASING_ITEM_WIDTH_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_INCREASING_ITEM_VALUE_STRIP_PACKER_FFDH.csv')
    fix_upper_bound('BASELINE_SORTED_INCREASING_ITEM_RATIO_STRIP_PACKER_FFDH.csv')

    return all_results


def load_split_results():
    all_results = {}

    def load_results(fn, df):
        fn = fn.replace(".csv", "")
        fn = fn.replace("SPLIT_BASELINE-SORTED_", "")
        fn = fn.replace("ITEM_", "")
        fn = fn.replace("-BOTTOM_LEFT", "")
        fn = fn.replace("DECREASING_", '-')
        fn = fn.replace("INCREASING_", '+')
        all_results[fn.lower()] = df

    def fix_upper_bound(fn):
        df = pd.read_csv(f'../split_csvs/{fn}')
        names = df['instance']

        for name in names:
            value = UB.loc[UB['instance'] == name, 'upper bound'].values[0]
            df.loc[df['instance'] == name, 'upper bound'] = value

        # df.to_csv(f'../fixed/{fn}')
        load_results(fn, df)
        return df

    fix_upper_bound('SPLIT_BASELINE-SORTED_DECREASING_ITEM_HEIGHT-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_DECREASING_ITEM_WIDTH-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_DECREASING_ITEM_VALUE-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_DECREASING_ITEM_RATIO-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_INCREASING_ITEM_HEIGHT-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_INCREASING_ITEM_WIDTH-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_INCREASING_ITEM_VALUE-BOTTOM_LEFT.csv')
    fix_upper_bound('SPLIT_BASELINE-SORTED_INCREASING_ITEM_RATIO-BOTTOM_LEFT.csv')
    return all_results


def find_best_lower_bound(df):
    bestScore = {}

    for instance in instances_names:
        all_scores = {}
        for algorithm in df:
            algo_results = df[algorithm]
            instance_mask = algo_results['instance'] == instance
            algo_score = algo_results.loc[instance_mask, 'total score']
            all_scores[algorithm] = algo_score.values

        # print(instance, all_scores)
        sorted_scores = {k: v for k, v in sorted(all_scores.items(), key=lambda i: i[1], reverse=True)}

        best = next(iter(sorted_scores.items()))
        bestScore[instance] = best
        # print(f'Results for instance{instance}: algorithm = {best[0]}, score = {best[1]}')

    return bestScore


def find_best_total_score_splits(df):
    bestScore = {}
    for instance in split_names:
        all_scores = {}
        for algorithm in df:
            algo_results = df[algorithm]
            instance_mask = algo_results['instance'] == instance
            algo_score = algo_results.loc[instance_mask, 'total score']
            all_scores[algorithm] = algo_score.values

        # print(instance, all_scores)
        sorted_scores = {k: v for k, v in sorted(all_scores.items(), key=lambda i: i[1], reverse=True)}

        best = next(iter(sorted_scores.items()))
        bestScore[instance] = best
        # print(f'Results for instance{instance}: algorithm = {best[0]}, score = {best[1]}')
    return bestScore


def update_lower_bound(df, best_lb):
    df['lb algorithm'] = pd.Series(index=df.index)
    for instance in instances_names:
        score = best_lb[instance]
        instance_mask = df['instance'] == instance
        df.loc[instance_mask, 'lower bound'] = score[1]
        df.loc[instance_mask, 'lb algorithm'] = score[0]
    return df


def update_score_splits(df, best_sc):
    df['best algorithm'] = pd.Series(index=df.index)
    for instance in split_names:
        score = best_sc[instance]
        instance_mask = df['instance'] == instance
        df.loc[instance_mask, 'best score'] = score[1]
        df.loc[instance_mask, 'best algorithm'] = score[0]
    return df


def extract_features_metrics(fdf, lr):
    result_cols = ['num i. packed', 'total score', 'improvement', 'loss', 'coverage', 'utilization', 'ms']
    features_cols = fdf.columns.drop(result_cols)
    ff = fdf[features_cols]
    lrm = {}
    extract = lambda d: d[['instance'] + result_cols]

    for algorithm in lr:
        lrm[algorithm] = extract(lr[algorithm])

    return ff, lrm


def compare_column_over_algorithms(cols, lrm):
    dfs = []
    for algorithm, result_metrics in lrm.items():
        subset = result_metrics[['instance'] + cols].set_index('instance')
        subset = subset.rename(columns={col: f"{algorithm}_{col}" for col in cols})
        dfs.append(subset)

    final_result = pd.concat(dfs, axis=1)
    return final_result


def gather_result_with_features(fdf, f_col, lrm, r_col):
    combined = {}
    for algorithm, results in lrm.items():
        result_cols = results[['instance'] + r_col]
        feature_cols = fdf[['instance'] + f_col]
        combined[algorithm] = pd.merge(feature_cols, result_cols, on='instance')

    return combined


def filter_type(df, instance_type):
    return df[df['type'] == instance_type.upper()]


def add_individual_result_variable(df, lrm, var):
    for algorithm, result in lrm.items():
        df[f'{algorithm}'] = result[var]
    return df


def gather_type_with_utilization_and_algorithm(df, lrm):
    mdf = []
    it = df[['instance', 'type']]

    for algorithm, result in lrm.items():
        utilization = result[['instance', 'utilization']]
        utilization = pd.merge(it, utilization, on='instance')
        utilization['algorithm'] = f'{algorithm}'
        mdf.append(utilization)
    return pd.concat(mdf, axis=0)


def gather_type_with_features_and_algorithm(df, lrm):
    mdf = []
    it = df[['instance', 'type', 'area c.', 'sum i. value']]

    for algorithm, result in lrm.items():
        utilization = result[['instance', 'coverage', 'utilization', 'total score']]
        # utilization.loc['coverage ratio'] = utilization['coverage'] / it['area c.']
        utilization = pd.merge(it, utilization, on='instance')
        utilization['algorithm'] = f'{algorithm}'
        mdf.append(utilization)

    return pd.concat(mdf, axis=0)


def plot_utilization_type(df, lrm):
    merged = gather_type_with_utilization_and_algorithm(df, lrm)
    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(right=0.78)
    sns.boxplot(merged, x='type', y='utilization', hue='algorithm', linewidth=0.5, fliersize=1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Algorithm utilization - instance types')
    plt.show()


def plot_utilization_coverage_kde(df, lrm):
    merged = gather_type_with_features_and_algorithm(df, lrm)
    area_coverage = merged['coverage'] / merged['area c.']
    value_coverage = merged['total score'] / merged['sum i. value']
    print(area_coverage)
    plt.figure(figsize=(6, 4))
    sns.jointplot(merged, x=value_coverage, y=area_coverage, kind='kde', hue='type', palette='tab10',
                  levels=10, fill=False, thresh=.15)
    plt.suptitle('KDE-utilization - coverage instance types')
    plt.show()


def generate_lb_plot():
    loaded_results = load_fixed_ub_results()           # load all results

    fixed_algorithm = '-height'  # choose one since features should be the same
    fixed_dataset = loaded_results[fixed_algorithm]        # dataset from fixed algorithm
    best_lb_scores = find_best_lower_bound(loaded_results)            # find best score to determine the best lower bound
    fixed_dataset = update_lower_bound(fixed_dataset, best_lb_scores)
    fixed_features, loaded_results_metrics = extract_features_metrics(fixed_dataset, loaded_results)

    # merged = compare_column_over_algorithms(['utilization', 'coverage'], loaded_results_metrics)
    # area_container = fixed_features[['instance', 'area c.']].set_index('instance')
    # merged_utilization = gather_result_with_features(fixed_features, ['type', 'sum i. value'], loaded_results_metrics, ['utilization', 'lb algorithm'])
    # all_scores = add_individual_result_variable(fixed_dataset, loaded_results_metrics, 'utilization')

    plot_utilization_type(fixed_dataset, loaded_results_metrics)


def make_splits_csv():
    loaded_splits = load_split_results()
    fixed_splits = loaded_splits['-height']
    best_scores = find_best_total_score_splits(loaded_splits)            # find best score to determine the best lower bound
    fixed_data_splits = update_score_splits(fixed_splits, best_scores)
    print(fixed_data_splits)
    fixed_data_splits.to_csv('../split_csvs/final.csv')


def generate_kde_lb_plot():
    loaded_results = load_fixed_ub_results()  # load all results
    fixed_dataset = loaded_results['-height']  # dataset from fixed algorithm
    best_lb_scores = find_best_lower_bound(loaded_results)  # find best score to determine the best lower bound
    fixed_dataset = update_lower_bound(fixed_dataset, best_lb_scores)
    fixed_features, loaded_results_metrics = extract_features_metrics(fixed_dataset, loaded_results)
    plot_utilization_coverage_kde(fixed_dataset, loaded_results_metrics)


generate_kde_lb_plot()
