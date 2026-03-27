import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

def thin_points(date_group: pd.DataFrame, area: int) -> pd.DataFrame:
    """
    inspect groups of rows that come from the same date but different locations.
    if locations are within [area] meters of each other, keep only one row in that area.
    """

    # if the group only contains 1 sample, do nothing
    if len(date_group) < 2:
        return date_group
    
    temp_group = date_group.reset_index(drop=True)
        
    # convert latitude and longitude to radians
    coords = np.deg2rad(temp_group[['lat_fixed', 'lon_fixed']].values)

    # build a ball tree using haversine metric for distance (accounts for spherical earth)
    tree = BallTree(coords, metric='haversine')
    radians_area = area / 6371000.0

    # find neighboring points for each point in the group
    indices = tree.query_radius(coords, r=radians_area)

    already_covered = set()
    keep_indices = []
    
    for i, neighbors in enumerate(indices):     # loop through each point in the group
        if i not in already_covered:            # if the point has not been covered:
            keep_indices.append(i)                  # keep it to represent the neighbor group
            already_covered.update(neighbors)       # add all of its neighbors to already_covered
    
    return temp_group.iloc[keep_indices].copy()

def load_clean_data(path: str) -> pd.DataFrame:

    # load data
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Data loading unsuccessful: {e}")
        return

    print("rows pre-cleaning:", df.shape[0])

    # remove rows where sensor is NONE
    print("rows removed due to no sensor match:", df[df["sensor"]=="NONE"].shape[0])
    df = df[df["sensor"]!="NONE"]

    # remove rows with impossible band values 
    band_cols = ["B2", "B3", "B4", "B5"]
    valid_bands = ((df[band_cols] > 0) & (df[band_cols] < 10000)).all(axis=1)
    print("rows removed due to impossible band values:", df[~valid_bands].shape[0])
    df = df[valid_bands].copy()

    # thin data by keeping 1 point for each 50m area on the same day
    rows_before_thinning = df.shape[0]
    thinned_groups = []
    for _, group in df.groupby('date', sort=False):  # group by date
        thinned = thin_points(group, 50)                # thin each group
        thinned_groups.append(thinned)                  # add to thinned groups list
    df = pd.concat(thinned_groups, ignore_index=True)   # turn back into df
    print("rows removed due to spatial overlap:", rows_before_thinning - df.shape[0])

    # sale band values
    df[band_cols] = df[band_cols] / 10000

    # scale abundance values
    df['log_abun'] = np.log10(df['abun'] + 1)

    # turn dates into datetime objects
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", yearfirst=True)
    df["sat_date"] = pd.to_datetime(df["sat_date"], format="%Y-%m-%d", yearfirst=True)

    # create column for difference in days between sample and sensor measurement
    df["diff_days"] = (df["date"] - df["sat_date"]).dt.days.abs()

    # encode sensor column and remove string sensor column
    df["is_s2"] = (df["sensor"]=="S2").astype(int)
    df.drop(columns="sensor", inplace=True)

    print("rows post-cleaning:", df.shape[0])
    return df

if __name__ == "__main__":

    data_path = "data/CAML_Algae_Final.csv"
    df = load_clean_data(data_path)