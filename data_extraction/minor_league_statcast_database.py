import pandas as pd
from datetime import datetime, timedelta
import sqlite3

from sc_search_minor_leagues import statcast_minor_leagues

#initial connection to statcast DB
conn = sqlite3.connect('minor_league_statcast_data.db')

#acquire opening day data for testings
od = datetime(2025, 3, 28)
od_str = od.strftime('%Y-%m-%d')
df_opening_day = statcast_minor_leagues(start_dt=od_str,end_dt=od_str)

# Print unique home teams
print("Unique Home Teams:")
print(df_opening_day['home_team'].unique())

# Print unique away teams
print("\nUnique Away Teams:")
print(df_opening_day['away_team'].unique())