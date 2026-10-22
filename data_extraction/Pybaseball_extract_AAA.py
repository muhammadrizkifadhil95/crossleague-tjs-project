import pandas as pd
import numpy as np
from pybaseball import statcast, playerid_reverse_lookup
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import os
import pybaseball
import calendar

pybaseball.cache.enable()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data')

HAND_PATH = os.path.join(DATA_DIR, 'pitcher_hand.csv')
TJ_PATH = os.path.join(DATA_DIR, 'list of TJ.csv')
BIO_PATH = os.path.join(DATA_DIR, 'biofile0.csv')

# -----------------------------
# 1. AAA Data Fetch [CUSTOM ENGINE]
# -----------------------------
from sc_search_minor_leagues import statcast_minor_leagues

# Path for AAA chunks
chunk_dir = './data/aaa_chunks'
os.makedirs(chunk_dir, exist_ok=True)

# Window: 2023-2025 (Full AAA Statcast availability)
years = [2023, 2024, 2025]
months = range(3, 10)


def fetch_aaa_month(year, month):
    file_path = f"{chunk_dir}/aaa_{year}_{month:02d}.csv"
    if os.path.exists(file_path):
        return

    start_dt = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end_dt = f"{year}-{month:02d}-{last_day}"

    try:
        print(f"Fetching AAA {year}-{month:02d}...")
        # Custom engine call
        month_df = statcast_minor_leagues(start_dt=start_dt, end_dt=end_dt)
        if month_df is not None and not month_df.empty:
            month_df.to_csv(file_path, index=False)
            del month_df
    except Exception as e:
        print(f"Error at AAA {year}-{month:02d}: {e}")


for y in years:
    for m in months:
        fetch_aaa_month(y, m)

# -----------------------------
# 2. Data Filtering
# -----------------------------
all_aaa_files = sorted([f"{chunk_dir}/{f}" for f in os.listdir(chunk_dir) if f.endswith('.csv')])

select_columns = [
    'player_name', 'game_type', 'home_team', 'pitch_type', 'game_date', 'p_throws',
    'pitcher', 'release_speed', 'release_pos_x', 'release_pos_z', 'game_year',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'effective_speed', 'release_spin_rate', 'release_extension', 'spin_axis'
]

processed_chunks = []
if all_aaa_files:
    print(f"Consolidating {len(all_aaa_files)} AAA chunks...")
    for f in tqdm(all_aaa_files):
        temp_chunk = pd.read_csv(f)
        # Match MLB Regular Season filter
        temp_chunk = temp_chunk[temp_chunk['game_type'] == 'R']

        # Select columns available in AAA Statcast
        existing_cols = [c for c in select_columns if c in temp_chunk.columns]
        temp_chunk = temp_chunk[existing_cols]
        processed_chunks.append(temp_chunk)

    data = pd.concat(processed_chunks, ignore_index=True)
    data['game_date'] = pd.to_datetime(data['game_date'])
    data = data.sort_values('game_date').reset_index(drop=True)
    data_selected = data.astype('object').replace({pd.NA: np.nan})
    del processed_chunks

# -----------------------------
# 3. Pivot Table (Averaging Pitch Attributes by Pitch Type)
# -----------------------------
aggregations = {
    'release_speed': 'mean',
    'release_pos_x': 'mean',
    'release_pos_z': 'mean',
    'pfx_x': 'mean',
    'pfx_z': 'mean',
    'plate_x': 'mean',
    'plate_z': 'mean',
    'vx0': 'mean',
    'vy0': 'mean',
    'vz0': 'mean',
    'ax': 'mean',
    'ay': 'mean',
    'az': 'mean',
    'effective_speed': 'mean',
    'release_spin_rate': 'mean',
    'release_extension': 'mean',
    'spin_axis': 'mean'
}

pivot_data = data_selected.pivot_table(
    index=['player_name', 'home_team', 'game_date', 'pitcher', 'game_year'],
    columns=['pitch_type'],
    values=list(aggregations.keys()),
    aggfunc='mean'
)

pivot_data.reset_index(inplace=True)

# Rename pivoted columns (flatten MultiIndex)
pivot_data.columns = [
    '_'.join(col).strip() if col[1] else col[0]
    for col in pivot_data.columns.values
]

# -----------------------------
# 4. Data Quality Check
#    (Remove columns with >90% missing OR only one unique value)
# -----------------------------
final_join_pre = pivot_data.astype('object').replace({pd.NA: np.nan})

col = []
missing = []
level = []
for name in final_join_pre.columns:
    missper = final_join_pre[name].isnull().sum() / final_join_pre.shape[0]
    missing.append(round(missper, 4))
    lel = final_join_pre[name].dropna()
    level.append(len(list(set(lel))))
    col.append(name)

summary = pd.concat([
    pd.DataFrame(col, columns=['name']),
    pd.DataFrame(missing, columns=['Missing Percentage']),
    pd.DataFrame(level, columns=['Level'])
], axis=1)

summary['Missing Percentage'] = pd.to_numeric(summary['Missing Percentage'], errors='coerce')
drop_col = summary['name'][
    (summary['Level'] <= 1) | (summary['Missing Percentage'] >= 0.90)
    ]
final_join_remove = final_join_pre.drop(columns=drop_col)

# -----------------------------
# 5. Merge with Tommy John Surgery Data
# -----------------------------
tj_list = pd.read_csv(TJ_PATH)
tj_list_join = tj_list[['mlbamid', 'tj_surgery_date', 'Year']]
tj_list_join = tj_list_join.sort_values(by=['mlbamid', 'Year'])
tj_list_join['count'] = tj_list_join.groupby('mlbamid').cumcount() + 1

# Pivot to handle multiple surgeries per pitcher
tj_list_pivot = tj_list_join.pivot_table(
    index='mlbamid',
    columns='count',
    values=['tj_surgery_date', 'Year'],
    aggfunc=lambda x: ' '.join(map(str, x))
)
tj_list_pivot.columns = [
    '_'.join([str(i) for i in col]).strip()
    for col in tj_list_pivot.columns.values
]
tj_list_pivot.columns = [col.replace('.0', '') for col in tj_list_pivot.columns]
tj_list_pivot.reset_index(inplace=True)

# Merge final data with pivoted Tommy John data
result = pd.merge(
    final_join_remove,
    tj_list_pivot,
    left_on='pitcher',
    right_on='mlbamid',
    how='left'
).drop(columns='mlbamid')

# Create columns for multiple Tommy John surgery checks
result['tj_surgery_date_1'] = pd.to_datetime(result['tj_surgery_date_1'])
result['tj_surgery_date_2'] = pd.to_datetime(result['tj_surgery_date_2'])
result['tj_surgery_date_3'] = pd.to_datetime(result['tj_surgery_date_3'])
result['TJ Surgery Year_1'] = result['tj_surgery_date_1'].dt.year
result['TJ Surgery Year_2'] = result['tj_surgery_date_2'].dt.year
result['TJ Surgery Year_3'] = result['tj_surgery_date_3'].dt.year
result['season_before_tj_1'] = result['TJ Surgery Year_1'] - result['game_year']
result['season_before_tj_2'] = result['TJ Surgery Year_2'] - result['game_year']
result['season_before_tj_3'] = result['TJ Surgery Year_3'] - result['game_year']

# Calculate date differences
result['tj_surgery_date_1'] = pd.to_datetime(result['tj_surgery_date_1'], errors='coerce')
result['tj_surgery_date_2'] = pd.to_datetime(result['tj_surgery_date_2'], errors='coerce')
result['tj_surgery_date_3'] = pd.to_datetime(result['tj_surgery_date_3'], errors='coerce')
result['game_date'] = pd.to_datetime(result['game_date'], errors='coerce')
result['diff_date_1'] = (result['tj_surgery_date_1'] - result['game_date']).dt.days
result['diff_date_2'] = (result['tj_surgery_date_2'] - result['game_date']).dt.days
result['diff_date_3'] = (result['tj_surgery_date_3'] - result['game_date']).dt.days


def get_closest_surgery_date(row):
    """
    Returns the closest valid (diff > 0) tj_surgery_date among
    up to three possible surgeries.
    """
    diffs_dates = [
        (row['diff_date_1'], row['tj_surgery_date_1']),
        (row['diff_date_2'], row['tj_surgery_date_2']),
        (row['diff_date_3'], row['tj_surgery_date_3'])
    ]
    valid_dates = [date for diff, date in diffs_dates if diff > 0]
    if valid_dates:
        return min(valid_dates)
    return pd.NaT


result['tj_surgery_date'] = result.apply(get_closest_surgery_date, axis=1)
result['TJ Surgery Year'] = result['tj_surgery_date'].dt.year


# Combine 'season_before_tj' columns to find the minimum valid non-negative
def min_non_negative(row):
    values = [
        row['season_before_tj_1'],
        row['season_before_tj_2'],
        row['season_before_tj_3']
    ]
    valid_values = [v for v in values if v >= 0]
    return min(valid_values) if valid_values else pd.NA


result['season_before_tj'] = result.apply(min_non_negative, axis=1)

# -----------------------------
# 6. Condition-Based Filtering
#    (Identify pitchers who meet certain Tommy John surgery conditions)
# -----------------------------
df = result.copy()


def check_conditions(group):
    """
    1) condition_1: The pitcher has a TJ surgery season (season_before_tj == 0)
       and at least 2 unique 'season_before_tj' records > 0 before that season.
    2) condition_2: No TJ surgery season in the data, but at least 3 unique
       'season_before_tj' > 0.
    Exclude if min_season_before_tj >= 2 for the group.
    """
    group['condition_1'] = False
    group['condition_2'] = False
    group['exclude'] = False

    min_season_before_tj = group['season_before_tj'].min()
    if min_season_before_tj >= 2:
        group['exclude'] = True
        return group

    tj_surgery_season = group[group['season_before_tj'] == 0]
    has_tj_surgery_season = not tj_surgery_season.empty
    pre_surgery_seasons_count = len(set(group[group['season_before_tj'] > 0]['season_before_tj']))

    if has_tj_surgery_season and pre_surgery_seasons_count >= 2:
        group['condition_1'] = True
    if not has_tj_surgery_season and pre_surgery_seasons_count >= 3:
        group['condition_2'] = True
    return group


df = df.groupby(['pitcher', 'TJ Surgery Year'], dropna=False).apply(check_conditions)
df['condition'] = df['condition_1'] | df['condition_2']

# Filter out excluded pitchers
filtered_result = df[(df['condition'] | pd.isna(df['tj_surgery_date'])) & ~df['exclude']]

# Remove data beyond 'season_before_tj' >= 4
filtered_result = filtered_result[
    (filtered_result['season_before_tj'] < 4) | (pd.isna(filtered_result['season_before_tj']))
    ].reset_index(drop=True)

# -----------------------------
# 7. Handedness Adjustment
#    (Flip certain columns if pitcher is LHP)
# -----------------------------
# We reach back to 'data' or 'data_selected' from Phase 2 if 'p_throws' is missing here
source_df = filtered_result
if 'p_throws' not in source_df.columns:
    if 'data_selected' in locals():
        source_df = locals()['data_selected']
    elif 'data' in locals():
        source_df = locals()['data']

# Now the mapping will work because it's pulling from the raw Phase 2 data
if 'p_throws' in source_df.columns:
    hand_map = (
        source_df[['pitcher', 'p_throws']]
        .dropna()
        .drop_duplicates('pitcher')
        .set_index('pitcher')['p_throws']
        .to_dict()
    )
    # Apply that map to our current filtered_result
    result_hand = filtered_result.copy()
    result_hand['hand'] = result_hand['pitcher'].map(hand_map).map({'L': 'lhp', 'R': 'rhp'})
else:
    print("Warning: p_throws not found in Phase 2 or Phase 6. Defaulting to rhp.")
    result_hand = filtered_result.copy()
    result_hand['hand'] = 'rhp'


def adjust_for_hand(row):
    """
    Flip the sign of certain pitch-related columns for LHP
    and adjust spin axis for left-handers.
    """
    for col in row.index:
        if 'ax_' in col or 'pfx_x_' in col or 'release_pos_x_' in col or 'vx0_' in col:
            if row['hand'] == 'lhp':
                row[col] = -row[col]
        if 'spin_axis_' in col:
            if row['hand'] == 'lhp':
                # 360 - spin_axis for left-handers
                row[col] = 360 - row[col] if pd.notnull(row[col]) else np.nan
    return row


filtered_result_hand = result_hand.apply(adjust_for_hand, axis=1)

# -----------------------------
# 8. Time-to-Surgery Features
# -----------------------------
filtered_result_hand['game_date'] = pd.to_datetime(filtered_result_hand['game_date'])
filtered_result_hand['tj_surgery_date'] = pd.to_datetime(filtered_result_hand['tj_surgery_date'], errors='coerce')

# Calculate days before TJ, mark out-of-range with 9999
filtered_result_hand['days_before_tj'] = (
        filtered_result_hand['tj_surgery_date'] - filtered_result_hand['game_date']
).dt.days
filtered_result_hand['days_before_tj'].fillna(9999, inplace=True)
filtered_result_hand['days_before_tj'] = np.where(
    (filtered_result_hand['days_before_tj'] < 0) | (filtered_result_hand['days_before_tj'] > 1290),
    9999,
    filtered_result_hand['days_before_tj']
)

# Minimum days_before_tj per pitcher
filtered_result_hand['min_days_before_tj'] = (
    filtered_result_hand.groupby('pitcher')['days_before_tj'].transform('min')
)


def adjust_days(df):
    """
    Create 'adjust_days_before_tj' based on the difference
    from each pitcher's minimum non-9999 days_before_tj.
    """
    df['adjust_days_before_tj'] = np.where(
        (df['days_before_tj'] == 9999),
        9999,
        df['days_before_tj'] - df['min_days_before_tj']
    )
    return df


final_result = adjust_days(filtered_result_hand)

# Remove one specific player
final_result = final_result[final_result['player_name'] != 'Axford, John']
final_result.sort_values(by=['player_name', 'game_date'], inplace=True)

# -----------------------------
# 9. Fill Missing Pitch-Type Columns via Interpolation
# -----------------------------
df = final_result.copy()

columns_to_calculate = [
    'ax_CH', 'ax_CU', 'ax_FC', 'ax_FF', 'ax_SI', 'ax_SL',
    'ay_CH', 'ay_CU', 'ay_FC', 'ay_FF', 'ay_SI', 'ay_SL',
    'az_CH', 'az_CU', 'az_FC', 'az_FF', 'az_SI', 'az_SL',
    'effective_speed_CH', 'effective_speed_CU', 'effective_speed_FC', 'effective_speed_FF',
    'effective_speed_SI', 'effective_speed_SL',
    'pfx_x_CH', 'pfx_x_CU', 'pfx_x_FC', 'pfx_x_FF', 'pfx_x_SI', 'pfx_x_SL',
    'pfx_z_CH', 'pfx_z_CU', 'pfx_z_FC', 'pfx_z_FF', 'pfx_z_SI', 'pfx_z_SL',
    'plate_x_CH', 'plate_x_CU', 'plate_x_FC', 'plate_x_FF', 'plate_x_SI', 'plate_x_SL',
    'plate_z_CH', 'plate_z_CU', 'plate_z_FC', 'plate_z_FF', 'plate_z_SI', 'plate_z_SL',
    'release_extension_CH', 'release_extension_CU', 'release_extension_FC', 'release_extension_FF',
    'release_extension_SI', 'release_extension_SL',
    'release_pos_x_CH', 'release_pos_x_CU', 'release_pos_x_FC', 'release_pos_x_FF',
    'release_pos_x_SI', 'release_pos_x_SL',
    'release_pos_z_CH', 'release_pos_z_CU', 'release_pos_z_FC', 'release_pos_z_FF',
    'release_pos_z_SI', 'release_pos_z_SL',
    'release_speed_CH', 'release_speed_CU', 'release_speed_FC', 'release_speed_FF',
    'release_speed_SI', 'release_speed_SL',
    'release_spin_rate_CH', 'release_spin_rate_CU', 'release_spin_rate_FC',
    'release_spin_rate_FF', 'release_spin_rate_SI', 'release_spin_rate_SL',
    'spin_axis_CH', 'spin_axis_CU', 'spin_axis_FC', 'spin_axis_FF', 'spin_axis_SI', 'spin_axis_SL',
    'vx0_CH', 'vx0_CU', 'vx0_FC', 'vx0_FF', 'vx0_SI', 'vx0_SL',
    'vy0_CH', 'vy0_CU', 'vy0_FC', 'vy0_FF', 'vy0_SI', 'vy0_SL',
    'vz0_CH', 'vz0_CU', 'vz0_FC', 'vz0_FF', 'vz0_SI', 'vz0_SL'
]

# Interpolate or fill with global mean if all values are NaN for that pitcher
total_tasks = len(df['pitcher'].unique()) * len(columns_to_calculate)
progress_bar = tqdm(total=total_tasks, desc="Processing", unit="task")

for pitcher in df['pitcher'].unique():
    pitcher_data = df[df['pitcher'] == pitcher]

    for column in columns_to_calculate:
        if pitcher_data[column].isnull().all():
            # REFINEMENT: Fill with the mean of the specific game_year, not the global mean
            # We iterate through the years this specific pitcher appeared in
            for year in pitcher_data['game_year'].unique():
                yearly_mean = df[df['game_year'] == year][column].mean()

                # If the entire league had no data for that pitch in that year (rare),
                # fallback to global mean
                if pd.isna(yearly_mean):
                    yearly_mean = df[column].mean()

                df.loc[(df['pitcher'] == pitcher) & (df['game_year'] == year), column] = yearly_mean
        else:
            # Keep original interpolation logic for pitchers who HAVE the pitch type
            df.loc[df['pitcher'] == pitcher, column] = pitcher_data[column].interpolate(
                method='linear',
                limit_direction='both'
            )
        progress_bar.update(1)

progress_bar.close()

# -----------------------------
# 10. Combine TJ and non-TJ Groups (AAA VERSION)
# -----------------------------
print("Calculating seasonal pitch counts for Exposure filter (700 pitch threshold)...")
# Calculate counts from raw 'data'
pitch_counts = data.groupby(['pitcher', 'game_year']).size().reset_index(name='season_pitch_count')

# Flag Workhorse seasons with the new 700-pitch limit
workhorse_seasons = pitch_counts[pitch_counts['season_pitch_count'] >= 700]

# Tenure: Still requiring >= 2 seasons in the 3-year window
tenure_counts = workhorse_seasons.groupby('pitcher').size().reset_index(name='workhorse_years')
eligible_control_ids = tenure_counts[tenure_counts['workhorse_years'] >= 2]['pitcher']

# Filter Healthy Control Group
no_tj_pitchers = df[df['TJ Surgery Year'].isnull()].copy()
consecutive_seasons_data = no_tj_pitchers[no_tj_pitchers['pitcher'].isin(eligible_control_ids)].copy()

print(f"Refined search identified {len(eligible_control_ids)} AAA control pitchers (>=700 pitches).")

# --- START OF DAY 0 ANCHOR FIX ---
# Anchor the AAA healthy group: find their last game and count backwards
# This ensures they survive the 5-day grouping/interpolation in Prepfortrain.py
consecutive_seasons_data['game_date'] = pd.to_datetime(consecutive_seasons_data['game_date'])
last_game_date_aaa = consecutive_seasons_data.groupby('pitcher')['game_date'].transform('max')
consecutive_seasons_data['new_before_tj'] = (last_game_date_aaa - consecutive_seasons_data['game_date']).dt.days
# --- END OF DAY 0 ANCHOR FIX ---

# Injury Group Selection with .loc to prevent SettingWithCopyWarning
injury_pitcher_data = df[df['adjust_days_before_tj'] < 1500].copy()
injury_pitcher_data.loc[:, 'new_before_tj'] = injury_pitcher_data['adjust_days_before_tj']

# Combine
merge_df = pd.concat([injury_pitcher_data, consecutive_seasons_data], ignore_index=True)
merge_df = merge_df[~merge_df['pitcher'].isin([458584])]

# -----------------------------
# 11. Drop Unused Columns
# -----------------------------
columns_to_drop = [
    'last_game_date', 'TJ Surgery Date_1', 'TJ Surgery Date_2', 'TJ Surgery Date_3',
    'Year of TJ_1', 'Year of TJ_2', 'Year of TJ_3',
    'TJ Surgery Year_1', 'TJ Surgery Year_2', 'TJ Surgery Year_3',
    'season_before_tj_1', 'season_before_tj_2', 'season_before_tj_3',
    'diff_date_1', 'diff_date_2', 'diff_date_3',
    'exclude', 'condition',
    # [FIX] Columns below may not exist if pitch_count pivot was never performed:
    # 'count_CH','count_CU','count_FC','count_FF','count_SI','count_SL',
    'season_before_tj', 'min_days_before_tj', 'adjust_days_before_tj',
    'hand', 'days_before_tj', 'home_team', 'age', 'game_date'
]
# [FIX] If any of these columns are not present, this drop may raise an error.
# Make sure these columns actually exist before dropping.
for col_to_drop in columns_to_drop:
    if col_to_drop in merge_df.columns:
        merge_df.drop(columns=col_to_drop, inplace=True)

# -----------------------------
# 12. Set Target (TJ vs. Non-TJ)
# -----------------------------
merge_df['target'] = np.where(merge_df['TJ Surgery Year'].notnull(), 1, 0)

# -----------------------------
# 13. Finalize Data for Classification (OPTIMIZED)
# -----------------------------
import requests

# Sort for consistent ordering
merge_df.sort_values(
    by=['player_name', 'target', 'new_before_tj'],
    inplace=True
)

# [OPTIMIZATION] Create copy and force de-fragmentation immediately
cal_diff_df = merge_df.copy()
cal_diff_df = cal_diff_df.copy()

columns_for_diff = [
    'ax_CH', 'ax_CU', 'ax_FC', 'ax_FF', 'ax_SI', 'ax_SL',
    'ay_CH', 'ay_CU', 'ay_FC', 'ay_FF', 'ay_SI', 'ay_SL',
    'az_CH', 'az_CU', 'az_FC', 'az_FF', 'az_SI', 'az_SL',
    'effective_speed_CH', 'effective_speed_CU', 'effective_speed_FC',
    'effective_speed_FF', 'effective_speed_SI', 'effective_speed_SL',
    'pfx_x_CH', 'pfx_x_CU', 'pfx_x_FC', 'pfx_x_FF', 'pfx_x_SI', 'pfx_x_SL',
    'pfx_z_CH', 'pfx_z_CU', 'pfx_z_FC', 'pfx_z_FF', 'pfx_z_SI', 'pfx_z_SL',
    'plate_x_CH', 'plate_x_CU', 'plate_x_FC', 'plate_x_FF', 'plate_x_SI', 'plate_x_SL',
    'plate_z_CH', 'plate_z_CU', 'plate_z_FC', 'plate_z_FF', 'plate_z_SI', 'plate_z_SL',
    'release_extension_CH', 'release_extension_CU', 'release_extension_FC',
    'release_extension_FF', 'release_extension_SI', 'release_extension_SL',
    'release_pos_x_CH', 'release_pos_x_CU', 'release_pos_x_FC', 'release_pos_x_FF',
    'release_pos_x_SI', 'release_pos_x_SL',
    'release_pos_z_CH', 'release_pos_z_CU', 'release_pos_z_FC', 'release_pos_z_FF',
    'release_pos_z_SI', 'release_pos_z_SL',
    'release_speed_CH', 'release_speed_CU', 'release_speed_FC', 'release_speed_FF',
    'release_speed_SI', 'release_speed_SL',
    'release_spin_rate_CH', 'release_spin_rate_CU', 'release_spin_rate_FC',
    'release_spin_rate_FF', 'release_spin_rate_SI', 'release_spin_rate_SL',
    'spin_axis_CH', 'spin_axis_CU', 'spin_axis_FC', 'spin_axis_FF', 'spin_axis_SI', 'spin_axis_SL',
    'vx0_CH', 'vx0_CU', 'vx0_FC', 'vx0_FF', 'vx0_SI', 'vx0_SL',
    'vy0_CH', 'vy0_CU', 'vy0_FC', 'vy0_FF', 'vy0_SI', 'vy0_SL',
    'vz0_CH', 'vz0_CU', 'vz0_FC', 'vz0_FF', 'vz0_SI', 'vz0_SL'
]

# Group by pitcher and target to get means
mean_df = cal_diff_df.groupby(['pitcher', 'target'])[columns_for_diff].mean().reset_index()
for col in columns_for_diff:
    mean_df.rename(columns={col: f'mean_{col}'}, inplace=True)

# [OPTIMIZATION] Consolidate after the merge to prevent fragmentation before the loop
cal_diff_df = pd.merge(cal_diff_df, mean_df, on=['pitcher', 'target'], how='left')
cal_diff_df = cal_diff_df.copy()

# Compute the difference from the group means
for col in columns_for_diff:
    cal_diff_df[f'diff_{col}'] = cal_diff_df[col] - cal_diff_df[f'mean_{col}']

# Remove specific players
cal_diff_df = cal_diff_df[~cal_diff_df['player_name'].isin(['Sobotka, Chad', 'Williams, Trevor'])]

# 1. Load the local bio source
print("Loading local biometric source (biofile0.csv)...")
bio_source = pd.read_csv(BIO_PATH, low_memory=False)
bio_source['search_name'] = bio_source['lastname'] + ', ' + bio_source['usename']
bio_source = bio_source.drop_duplicates(subset=['search_name'], keep='first')
bio_source = bio_source[['search_name', 'height', 'weight']]

# 2. Local Merge: Initial attach using player names
# Using how='left' ensures the AAA pitcher count remains stable
cal_diff_df = pd.merge(cal_diff_df, bio_source, left_on='player_name', right_on='search_name', how='left')

# 3. API Fallback: Filling missing biometrics via MLB Stats API
# This is critical for AAA as many prospects aren't in older bio files
missing_ids = cal_diff_df[cal_diff_df['height'].isna()]['pitcher'].unique()

if len(missing_ids) > 0:
    print(f"Attempting to rescue {len(missing_ids)} AAA pitchers via MLB Stats API...")
    for i in range(0, len(missing_ids), 50):
        batch = ",".join(map(str, [int(x) for x in missing_ids[i:i + 50]]))
        api_url = f"https://statsapi.mlb.com/api/v1/people?personIds={batch}"
        try:
            resp = requests.get(api_url, timeout=15).json()
            for person in resp.get('people', []):
                p_id = person['id']
                h_str = person.get('height', "0' 0\"")
                w = person.get('weight', 0)

                # Convert "6' 2\"" to total inches
                parts = h_str.replace('"', '').split("' ")
                if len(parts) == 2:
                    total_inches = (int(parts[0]) * 12) + int(parts[1])

                    # Update cal_diff_df directly
                    mask = cal_diff_df['pitcher'] == p_id
                    cal_diff_df.loc[mask, 'height'] = cal_diff_df.loc[mask, 'height'].fillna(total_inches)
                    cal_diff_df.loc[mask, 'weight'] = cal_diff_df.loc[mask, 'weight'].fillna(w)
        except Exception as e:
            print(f"API batch failed: {e}")

# 4. Final BMI Calculation
cal_diff_df['bmi'] = (cal_diff_df['weight'] * 0.453592) / ((cal_diff_df['height'] * 0.0254) ** 2)

# 5. Cleanup temporary columns
if 'search_name' in cal_diff_df.columns:
    cal_diff_df.drop(columns='search_name', inplace=True)

# 6. Audit for AAA data completeness
qualified_missing = cal_diff_df[cal_diff_df['height'].isna()][['player_name', 'pitcher']].drop_duplicates()
if not qualified_missing.empty:
    print(f"AAA AUDIT: {len(qualified_missing)} qualified AAA pitchers are still missing HW data.")
    qualified_missing.to_csv('data/audit_QUALIFIED_missing_biometrics_AAA.csv', index=False)
else:
    print("SUCCESS: All qualified AAA pitchers now have biometric data.")

# 7. Final Column Selection and Saving
available_cols = cal_diff_df.columns.tolist()
required_base = ['player_name', 'pitcher', 'new_before_tj', 'target']
biometric_cols = ['height', 'weight', 'bmi']

final_cols = (
        required_base +
        biometric_cols +
        [col for col in available_cols if col.startswith('diff_')]
)

final_df = cal_diff_df[final_cols].sort_values(by=['player_name', 'new_before_tj'])
final_df.to_csv('data/final_df_AAA.csv', index=False)
print(f"AAA Process complete. Final unique pitchers: {final_df['player_name'].nunique()}")

# -----------------------------
# 14. Save Final Preprocessed Data
# -----------------------------
final_df.to_csv(
    'data/final_df_AAA.csv',
    index=False
)