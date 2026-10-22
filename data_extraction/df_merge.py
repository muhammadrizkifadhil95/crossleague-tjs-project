import pandas as pd

# Load both finalized datasets
mlb_df = pd.read_csv('data/final_df_MLB.csv')
aaa_df = pd.read_csv('data/final_df_AAA.csv')

# Add a source tag (Optional, but helpful for informatics analysis)
mlb_df['league_level'] = 'MLB'
aaa_df['league_level'] = 'AAA'

# Merge the datasets
# They must have the same columns for this to work perfectly
final_master_df = pd.concat([mlb_df, aaa_df], ignore_index=True)

# Save the master file that will be fed into Prepfortrain.py
final_master_df.to_csv('data/final_df.csv', index=False)

print(f"Master Dataset Created.")
print(f"Total Rows: {len(final_master_df)}")
print(f"Total Target 1 (Injuries): {final_master_df['target'].sum()}")