import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

TREND_THRESHOLD = 0.10
ROLLING_WINDOW = 3
SHARP_TIME_THRESHOLD_DAYS = 7


def compute_region_labels(values):
    """
    Assigns each value into one of three regions (low, medium, high) using KMeans clustering 
    or unique values if too few exist.

    Steps:
    - Removes NaN values from input.
    - If fewer than 3 unique values exist → generates artificial centers to ensure 3 regions.
    - Otherwise → applies KMeans (n_clusters=3) to cluster values and extract sorted cluster centers.
    - For each value:
        * NaN → label = -1
        * <= smallest center → label = 0 (low)
        * <= middle center   → label = 1 (medium)
        * otherwise          → label = 2 (high)

    Parameters:
    - values (array-like): numeric sequence with possible NaNs.

    Returns:
    - labels (list[int]): region label for each value (-1 for NaN).
    - centers (list[float]): sorted list of 3 cluster centers.
    """

    clean_values = pd.Series(values).dropna().values
    if len(clean_values) == 0:
        return [0]*len(values), [0, 1, 2]

    data_reshaped = np.array(clean_values).reshape(-1, 1)

    if len(np.unique(clean_values)) < 3:
        centers = sorted(np.unique(clean_values))
        while len(centers) < 3:
            centers.append(centers[-1] + 1)
    else:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(data_reshaped)
        centers = sorted(kmeans.cluster_centers_.flatten())

    labels = []
    for v in values:
        if pd.isna(v):
            labels.append(-1)
        elif v <= centers[0]:
            labels.append(0)
        elif v <= centers[1]:
            labels.append(1)
        else:
            labels.append(2)
    return labels, centers


def check_trend_refactored(df, col_names):
    """
    Detects trends, anomalies, region shifts, and data consistency across grouped time-series data.

    Steps performed for each (Comp_Number, Point, Direction) group:
    1. Sorts data by Timestamp.
    2. For each column in `col_names`:
        - Trend detection:
            * Compares current value to rolling mean of previous values (window=3).
            * Labels upward trend (1), downward trend (-1), or stable (0) if relative change 
              exceeds ±10% threshold.
        - Anomaly scoring:
            * Computes z-score against historical values.
            * Normalizes score into [0,1] with cap at Z_MAX=3.5.
            * Classifies as "Normal", "Alarm", or "Danger".
        - Region labeling:
            * Uses `compute_region_labels` to cluster values into 3 discrete regions (0,1,2).
        - Sharp change detection:
            * Detects jumps between consecutive regions.
            * Full jump (0→2) within 7 days = sharp_score 1.0
            * Single jump (0→1 or 1→2) within 7 days = sharp_score 0.5
            * Otherwise = 0.0
    3. Consistency check:
        - Marks 'consistent'=1 if consecutive timestamps differ by ≤70 days, else 0.

    Parameters:
    - df (pd.DataFrame): input data with columns 
        ['Comp_Number', 'Point', 'Direction', 'Timestamp', <col_names>...].
    - col_names (list[str]): numeric column names to process.

    Returns:
    - df (pd.DataFrame): original dataframe with additional computed columns:
        * <col>_trend
        * <col>_trend_score
        * <col>_sharp_score
        * <col>_trend_condition
        * <col>_region
        * consistent
    """

    df = df.sort_values(by=['Comp_Number', 'Point',
                        'Direction', 'Timestamp']).reset_index(drop=True)
    df['consistent'] = 0

    grouped = df.groupby(['Comp_Number', 'Point', 'Direction'])
    Z_MAX = 3.5

    for col_name in col_names:
        trend_col = f'{col_name}_trend'
        trend_score_col = f'{col_name}_trend_score'
        sharp_col = f'{col_name}_sharp_score'
        condition_col = f'{col_name}_trend_condition'
        region_col = f'{col_name}_region'

        df[trend_col] = 0
        df[trend_score_col] = 0.0
        df[sharp_col] = 0.0
        df[condition_col] = ''
        df[region_col] = 0

        for name, group in grouped:
            group = group.sort_values('Timestamp')

            prev_avg = group[col_name].shift(1).rolling(
                window=ROLLING_WINDOW, min_periods=1).mean()
            change_ratio = (group[col_name] - prev_avg) / prev_avg
            trend = np.where(change_ratio >= TREND_THRESHOLD, 1,
                             np.where(change_ratio <= -TREND_THRESHOLD, -1, 0))
            df.loc[group.index, trend_col] = trend

            scores = []
            conditions = []
            for idx, val in zip(group.index, group[col_name]):
                history = group.loc[group.index < idx, col_name]
                if len(history) == 0:
                    scores.append(0)
                    conditions.append("No History")
                else:
                    mu = history.mean()
                    sigma = history.std(ddof=0)
                    z = (val - mu) / sigma if sigma != 0 else 0
                    score_norm = min(abs(z)/Z_MAX, 1.0)
                    scores.append(round(score_norm, 2))
                    if z <= 1:
                        conditions.append("Normal")
                    elif z <= 2:
                        conditions.append("Alarm")
                    else:
                        conditions.append("Danger")
            df.loc[group.index, trend_score_col] = scores
            df.loc[group.index, condition_col] = conditions

            region_labels, _ = compute_region_labels(group[col_name].values)
            df.loc[group.index, region_col] = region_labels

            prev_regions = [np.nan] + region_labels[:-1]
            prev_times = [pd.NaT] + list(group['Timestamp'].values[:-1])
            sharp_scores = []
            for r_curr, r_prev, t_curr, t_prev in zip(region_labels, prev_regions, group['Timestamp'], prev_times):
                if pd.isna(r_prev):
                    sharp_scores.append(0)
                    continue
                time_diff_days = (
                    t_curr - t_prev).days if pd.notna(t_prev) else None
                if r_curr - r_prev == 2 and time_diff_days is not None and time_diff_days <= SHARP_TIME_THRESHOLD_DAYS:
                    sharp_scores.append(1.0)
                elif r_curr - r_prev == 1 and time_diff_days is not None and time_diff_days <= SHARP_TIME_THRESHOLD_DAYS:
                    sharp_scores.append(0.5)
                else:
                    sharp_scores.append(0.0)
            df.loc[group.index, sharp_col] = sharp_scores

    for name, group in grouped:
        group = group.sort_values('Timestamp')
        time_diff = group['Timestamp'].diff().dt.days
        consistent = np.where((time_diff <= 70) & (time_diff.notna()), 1, 0)
        df.loc[group.index, 'consistent'] = consistent

    return df
