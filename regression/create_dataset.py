import ee
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import requests
import io

def extract_satellite_data(feature):
    """
    takes a feature (row) of the FeatureCollection of CAML data and matches it with satellite bands from S2 or L8
    """
    # get water sample date from caml dataset
    raw_date = ee.Number(feature.get("date")).format('%d')  # convert into right format for date
    sample_date = ee.Date.parse("YYYYMMdd", raw_date)

    # get water sample location
    geom = feature.geometry()
    coords = geom.coordinates()
    lat = coords.get(1)
    lon = coords.get(0)

    # create date window
    start = sample_date.advance(-5, "day")
    end = sample_date.advance(5, "day")

    # create a collection of potential s2 images based on location, dates, and cloud cover
    s2_collection = s2.filterBounds(geom) \
                      .filterDate(start, end) \
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)) \
                      .sort("CLOUDY_PIXEL_PERCENTAGE")
    
    # create a collection of potential l8 images based on location, dates, and cloud cover
    l8_collection = l8.filterBounds(geom) \
                      .filterDate(start, end) \
                      .filter(ee.Filter.lt("CLOUD_COVER", 50)) \
                      .sort("CLOUD_COVER")
    
    # get era5 image for temperature
    era5_img = era5.filterBounds(geom) \
                   .filterDate(sample_date, sample_date.advance(1, 'day')) \
                   .first()
    
    # select the best image 
    has_s2 = s2_collection.size().gt(0)
    has_l8 = l8_collection.size().gt(0)
    has_any = has_s2.Or(has_l8)
    best_img = ee.Image(ee.Algorithms.If(has_s2, s2_collection.first(), l8_collection.first()))
    sensor_type = ee.String(ee.Algorithms.If(has_s2, "s2", ee.Algorithms.If(has_l8, "l8", "NONE")))

    # get cloudiness
    cloud_cover = ee.Algorithms.If(
        has_s2, 
        best_img.get("CLOUDY_PIXEL_PERCENTAGE"), 
        ee.Algorithms.If(has_l8, best_img.get("CLOUD_COVER"), -999)
    )

    # get surface temp using l8 or era5 
    l8_temp = ee.Image(ee.Algorithms.If(has_l8,
                                        ee.Image(l8_collection.first()).select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15),
                                        ee.Image.constant(-999)))
    era5_temp = ee.Image(ee.Algorithms.If(era5_img, 
                                          ee.Image(era5_img).select("skin_temperature").subtract(273.15), 
                                          ee.Image.constant(-999)))
    temp_c = ee.Image(ee.Algorithms.If(has_l8, l8_temp, era5_temp)).rename("temp_c")
    
    # get satellite date
    sat_date = ee.Algorithms.If(has_any, ee.Date(best_img.get('system:time_start')).format('YYYY-MM-dd'), "NONE")

    # get and scale bands
    final_img = ee.Image(scale_bands(best_img, sensor_type)).addBands(temp_c)

    # average band values of pixels in the image to get 1 value for each 
    stats = final_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30)

    # create a template to make sure all of the columns always exist
    results_template = ee.Dictionary({
        "satellite_date": "NONE", "temp_c": -999, "cloud_cover": -999,
        "sensor": "NONE", "blue": -999, "green": -999, "red": -999, 
        "NIR": -999, "red_edge": -999, "SWIR": -999, "ndci": -999, "ndvi": -999, 
        "ndwi": -999, "ndti": -999
    })

    # create a dictionary with the actual data
    actual_data = ee.Dictionary({
        "longitude": lon,
        "latitude": lat,
        "satellite_date": sat_date,
        "sample_date": sample_date.format("YYYY-MM-dd"),
        "sensor": sensor_type,
        "cloud_cover": cloud_cover
    })

    # combine empty stats with actual (fills in missing data with -999)
    combined_stats = results_template.combine(stats, True).combine(actual_data, True)
    return feature.set(combined_stats)

def scale_bands(image, sensor_type):
    """
    get satellite bands of an image and scale them based on sensor type
    for s2: divide by 10,000
    for l8: multiply by .0000275 and subtract .2
    """
    s2_bands = image.select(["B2", "B3", "B4", "B5", "B8A", "B11"], 
                            ["blue", "green", "red", "NIR", "red_edge", "SWIR"]) \
                            .divide(10000)
    
    l8_bands = image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"], 
                            ["blue", "green", "red", "NIR", "SWIR"]) \
                            .multiply(0.0000275).subtract(0.2)
    l8_bands = l8_bands.addBands(ee.Image.constant(-999).rename("red_edge"))
    
    null_bands = ee.Image([-999, -999, -999, -999, -999, -999]).rename(["blue", "green", "red", "red_edge", "NIR", "SWIR"])

    # assign bands to image (if sensor type is none assign -999 to all)
    img = ee.Image(ee.Algorithms.If(ee.String(sensor_type).equals("s2"), s2_bands, 
                                    ee.Algorithms.If(ee.String(sensor_type).equals("l8"), l8_bands, null_bands)))

    # add calculated columns
    ndci = img.normalizedDifference(["red_edge", "red"]).rename("ndci") 
    ndvi = img.normalizedDifference(["NIR", "red"]).rename("ndvi") 
    ndwi = img.normalizedDifference(["green", "NIR"]).rename("ndwi")
    ndti = img.normalizedDifference(["green", "red"]).rename("ndti")
    
    return img.addBands([ndci, ndvi, ndwi, ndti]).set("is_s2", ee.Number(ee.Algorithms.If(ee.String(sensor_type).equals("s2"), 1, 0)))

def match_caml_satellite():
    """
    match entire CAML dataset to satellite images
    """
    # initialize the library
    ee.Initialize(project="ml4cc-final-project")

    # get caml dataset (actual cyanobacteria samples)
    caml = ee.FeatureCollection("projects/sat-io/open-datasets/HAB-DETECTION/CAML_cyanobacteria_abundance_20211229_R1")

    # get satellite datasets
    global s2
    global l8
    global era5
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

    # process data
    old_cols = ["latitude", "longitude", "abun", "sample_date", "satellite_date", "temp_c", "sensor", "blue", "green", "red", "red_edge", "NIR", "SWIR", "ndci", "ndvi", "ndwi", "ndti", "cloud_cover"]
    new_cols = ["latitude", "longitude", "cyanobacteria_abundance", "sample_date", "satellite_date", "temp_c", "sensor", "blue", "green", "red", "red_edge", "NIR", "SWIR", "ndci", "ndvi", "ndwi", "ndti", "cloud_cover"]

    processed_data = caml.filter(ee.Filter.gte("date", 20130101)).map(extract_satellite_data)

    # export to google drive
    export = processed_data.select(old_cols, new_cols)
    task = ee.batch.Export.table.toDrive(
        collection=export,
        description="caml_satellite_matchup",
        fileFormat="CSV",
    )  
    task.start()

def get_wqp(satellite_df_path):
    """
    match satellite images with wqp water nutrient data
    """
    # load data
    df = pd.read_csv(satellite_df_path)

    # round latitude and longitudes to 3 decimals (110m) 
    unique_locs = df[['latitude', 'longitude']].round(3).drop_duplicates()

    results_list = []

    # iterate through unique locations
    for _, loc in tqdm(unique_locs.iterrows(), total=len(unique_locs)):
        lat, lon = loc['latitude'], loc['longitude']
        
        # define 2km box around latitude and longitude
        bbox = f"{lon-0.02},{lat-0.02},{lon+0.02},{lat+0.02}"
        
        # construct url
        url = "https://www.waterqualitydata.us/data/Result/Search"
        params = {
            "bBox": bbox,
            "characteristicType": "Nutrient",
            "startDateLo": "01-01-2013",
            "startDateHi": "12-31-2021",
            "mimeType": "csv",
            "zip": "no"
        }
        
        # query
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # if the request worked, read csv content
            if response.status_code == 200 and len(response.content) > 1000: 
                data = pd.read_csv(io.StringIO(response.text), low_memory=False)
                data['query_lat'] = lat
                data['query_lon'] = lon
                results_list.append(data)
            
            time.sleep(0.5)
            
        except Exception as e:
            continue

    # if results found, turn results to csv 
    if results_list:
        nutrients = pd.concat(results_list, ignore_index=True)
        nutrients.to_csv('nutrients.csv', index=False)

    else:
        print("\nrequest successful, but no nutrients found in those specific boxes")

def normalize_units(row):
    """
    normalize nutrient units to mg/L
    """
    unit = str(row['ResultMeasure/MeasureUnitCode']).lower().strip()
    val = row['ResultMeasureValue']
    
    # mg/L or ppm - no change
    if 'mg/l' in unit or 'ppm' in unit or "mg n/l" in unit:

        try:
            return float(val)
        except:
            return None
    
    # micrograms - divide by 1000
    if 'ug/l' in unit or 'ppb' in unit:
        try:
            return float(val) / 1000.0
        except:
            return None
    
    # sediment/mass units - use as mg/L proxy
    if 'mg/kg' in unit or 'mg/g' in unit:
        try:
            return float(val)
        except:
            return None
        
    # otherwise return none
    return None

def match_wqp(existing_df_path, nutrient_df_path, date="sample"):
    """
    match wqp nutrient data with dates in the existing dataset
    """

    # load data
    df = pd.read_csv(existing_df_path)
    nutrients = pd.read_csv(nutrient_df_path)

    # normalize nutrient units and drop missing rows
    nutrients['ResultMeasureValue_normalized'] = nutrients.apply(normalize_units, axis=1)
    nutrients = nutrients.dropna(subset=['ResultMeasureValue_normalized'])

    # nutrient map grouping the types of nutrients into 4 groups
    nutrient_map = {
        # Phosphorus
        'Total Phosphorus, mixed forms': 'TP_mgL',
        'Phosphorus': 'TP_mgL',
        'Phosphorus, Particulate Organic': 'TP_mgL',
        'Organic phosphorus': 'TP_mgL',
        
        # Nitrogen
        'Total Nitrogen, mixed forms': 'TN_mgL',
        'Nitrogen': 'TN_mgL',
        'Total Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)': 'TN_mgL',
        'Nitrogen, mixed forms (NH3), (NH4), organic, (NO2) and (NO3)': 'TN_mgL',
        
        # Dissolved Inorganic Nitrogen
        'Nitrate + Nitrite': 'DIN_mgL',
        'Nitrate': 'DIN_mgL',
        'Nitrite': 'DIN_mgL',
        'Ammonia-nitrogen': 'DIN_mgL',
        'Ammonia': 'DIN_mgL',
        'Ammonium': 'DIN_mgL',
        
        # Orthophosphate
        'Orthophosphate': 'OrthoP_mgL',
        'Phosphate-phosphorus': 'OrthoP_mgL',
        'Soluble Reactive Phosphorus (SRP)': 'OrthoP_mgL'
    }

    # mapping the nutrients to their group names
    nutrients['type'] = nutrients['CharacteristicName'].map(nutrient_map)

    # pivot so each type group is a column rather than a row
    # use median if there are multiple samples in the same location and day
    nutrients_pivot = nutrients.pivot_table(
        index=['query_lat', 'query_lon', 'ActivityStartDate'], 
        columns='type', 
        values='ResultMeasureValue_normalized', 
        aggfunc='median'
    ).reset_index()

    # rename columns for clarity
    nutrients_pivot.columns.name = None 

    # add rounded coordinates to existing df
    df['lat_round'] = df['latitude'].round(3)
    df['lon_round'] = df['longitude'].round(3)

    # merge satellite data with the nutrient data, creating a row for every location match regardless of time
    merged = pd.merge(
        df, 
        nutrients_pivot, 
        left_on=['lat_round', 'lon_round'], 
        right_on=['query_lat', 'query_lon'], 
        how='left'
    )

    if date == "satellite":
        date_col = "satellite_date"
        days_diff_col = "days_diff_sat_to_nutrients"
    else:
        date_col = "caml_sample_date"
        days_diff_col = "days_diff_caml_to_nutrients"

    # calculate time difference between nutrient sample and satellite
    merged[date_col] = pd.to_datetime(merged[date_col])
    merged['ActivityStartDate'] = pd.to_datetime(merged['ActivityStartDate'])
    merged[days_diff_col] = (merged[date_col] - merged['ActivityStartDate']).dt.days.abs()

    nutrient_cols = ['TP_mgL', 'TN_mgL', 'DIN_mgL', 'OrthoP_mgL', 'ActivityStartDate']
    mask = merged[days_diff_col] > 14
    merged.loc[mask, nutrient_cols] = np.nan

    # filter for a 14-day window
    # final_matchups = merged[merged['days_diff_caml_to_nutrients'] <= 14]

    # fill missing phosphorus and nitrogen values with orthophosphate and nitrate/nitrite
    merged["TP_mgL"] = merged["TP_mgL"].fillna(merged["OrthoP_mgL"])
    merged["TN_mgL"] = merged["TN_mgL"].fillna(merged["DIN_mgL"])

    # rename columns
    merged.drop(columns={"DIN_mgL", "OrthoP_mgL", "lat_round", "lon_round", "query_lat", "query_lon"}, inplace=True)
    merged = merged.rename(columns={"ActivityStartDate": "nutrients_date"})

    # for each unique satellite image, drop all nutrient rows except the best one (least days difference)
    final_df = merged.sort_values(days_diff_col, ascending=True).drop_duplicates(
        subset=['latitude', 'longitude', date_col], 
        keep='first'
    )

    # save data
    final_df.to_csv("data/nutrients_matchup_timeseries.csv", index=False)

def match_nasa(existing_df_path):
    """
    match nasa weather data with dates and locations in the existing dataset
    """

    # load final matched dataframe
    df = pd.read_csv(existing_df_path)
    df['caml_sample_date'] = pd.to_datetime(df['caml_sample_date'])

    weather_results = []

    # iterate through unique location/date combos
    unique_matchups = df[['latitude', 'longitude', 'caml_sample_date']].drop_duplicates()

    print(f"fetching weather for {len(unique_matchups)} matchups...")

    # for each unique matchup, get weather data
    for _, row in tqdm(unique_matchups.iterrows(), total=len(unique_matchups)):

        lat, lon = row['latitude'], row['longitude']

        # convert dates for nasa power
        date_str = row['caml_sample_date'].strftime('%Y%m%d')
        
        # create window 3 days before so we can calculate accumulated precipitation
        start_date = (row['caml_sample_date'] - pd.Timedelta(days=3)).strftime('%Y%m%d')
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": date_str,
            "format": "JSON"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:

                # get data
                data = response.json()['properties']['parameter']
                
                # sum rain over the 3 days before
                precip_3d = sum(data['PRECTOTCORR'].values())

                # get wind and solar for the exact day
                wind = list(data['WS2M'].values())[-1]
                solar = list(data['ALLSKY_SFC_SW_DWN'].values())[-1]
                
                # add data to results
                weather_results.append({
                    'latitude': lat, 'longitude': lon, 'caml_sample_date': row['caml_sample_date'],
                    'precip_3d_mm': precip_3d,
                    'wind_speed_ms': wind,
                    'solar_rad_wm2': solar
                })
            
            else:
                print(f"failed: Status {response.status_code} at {lat}, {lon}")

            time.sleep(0.2) 
            
        except Exception as e:
            continue

    # merge weather back into main dataframe
    weather_df = pd.DataFrame(weather_results)
    df_final = pd.merge(df, weather_df, on=['latitude', 'longitude', 'caml_sample_date'], how='left')
    df_final.to_csv('final_model_ready_data.csv', index=False)

if __name__ == "__main__":
    
    match_caml_satellite()