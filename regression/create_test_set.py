import pandas as pd
import ee
from regression.create_train_set import scale_bands

def load_clean_ices(path: str) -> pd.DataFrame:
    
    # load data
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"Data loading unsuccessful: {e}")
        return

    print("shape pre-cleaning:", df.shape)

    # filter by cyanobacteria only
    df = df[df["PEG_division"] == "CYANOBACTERIA"].copy()

    # calculate cells per mL based on unit
    df["cells/mL"] = df["MUNIT"].case_when([
        (df["MUNIT"] == "mg/m3", (df["Value"] * df["PEG_cells_no_per_counting_unit"] * 1000) / df["PEG_volume_um3_per_counting_unit"]),
        (df["MUNIT"] == "ug/l", (df["Value"] * df["PEG_cells_no_per_counting_unit"] * 1000) / df["PEG_volume_um3_per_counting_unit"]),
        (df["MUNIT"] == "nrcells/l", df["Value"] / 1000),
        (df["MUNIT"] == "nr/l", (df["Value"] * df["PEG_cells_no_per_counting_unit"]) / 1000),
        (df["MUNIT"] == "nrfil100/m3", (df["Value"] * df["PEG_cells_no_per_counting_unit"]) / 1000000)
    ])

    # drop rows with nulls for cells/mL (had a null in one of the rows needed to calculate)
    df = df.dropna(subset=["cells/mL"])

    # group by date and location to get the total cyanobacteria at each location since multiple species of bacteria are in the data
    df_grouped = df.groupby(['Latitude', 'Longitude', 'DATE'])['cells/mL'].sum().to_frame(name="total cells/mL").reset_index().copy()
    df_grouped["total cells/mL"] = df_grouped["total cells/mL"].astype(float)

    print("shape post-cleaning:", df_grouped.shape)

    df_grouped.columns = df_grouped.columns.str.lower()
    return df_grouped

def extract_satellite_data(feature):

    # get latitude and longitude and create a point
    point = feature.geometry()
    coords = point.coordinates()
    longitude = ee.Number(coords.get(0))
    latitude = ee.Number(coords.get(1))

    # get date
    sample_date = ee.Date.parse("dd/MM/yyyy", ee.String(feature.get("date")))

    # create date window
    start = sample_date.advance(-7, "day")
    end = sample_date.advance(7, "day")

    # create a collection of potential s2 images based on location, dates, and cloud cover
    s2_collection = s2.filterBounds(point) \
                      .filterDate(start, end) \
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25)) \
                      .sort("CLOUDY_PIXEL_PERCENTAGE")
    
    # create a collection of potential l8 images based on location, dates, and cloud cover
    l8_collection = l8.filterBounds(point) \
                      .filterDate(start, end) \
                      .filter(ee.Filter.lt("CLOUD_COVER", 25)) \
                      .sort("CLOUD_COVER")
    
    # get era5 image for temperature
    era5_img = era5.filterBounds(point) \
                   .filterDate(sample_date, sample_date.advance(1, 'day')) \
                   .first()
    
    # select the best image 
    has_s2 = s2_collection.size().gt(0)
    has_l8 = l8_collection.size().gt(0)
    has_any = has_s2.Or(has_l8)
    best_img = ee.Image(ee.Algorithms.If(has_s2, s2_collection.first(), l8_collection.first()))
    sensor_type = ee.String(ee.Algorithms.If(has_s2, "s2", ee.Algorithms.If(has_l8, "l8", "NONE")))

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
    stats = final_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30)

    # create a template to make sure all of the columns always exist
    results_template = ee.Dictionary({
        "satellite_date": "NONE", "temp_c": -999, 
        "sensor": "NONE", "blue": -999, "green": -999, "red": -999, 
        "NIR": -999, "red_edge": -999, "SWIR": -999, "ndci": -999, "ndvi": -999, 
        "ndwi": -999, "ndti": -999
    })

    # create a dictionary with the actual data
    actual_data = ee.Dictionary({
        "longitude": longitude,
        "latitude": latitude,
        "satellite_date": sat_date,
        "sample_date": sample_date.format("YYYY-MM-dd"),
        "sensor": sensor_type
    })

    # combine empty stats with actual (fills in missing data with -999)
    combined_stats = results_template.combine(stats, True).combine(actual_data, True)
    return feature.set(combined_stats)

if __name__ == "__main__":
    
    ee.Initialize(project="ml4cc-final-project")

    # get ices dataset
    ices = ee.FeatureCollection("projects/ml4cc-final-project/assets/test_samples")

    # get satellite datasets
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

    final_cols = [
        "latitude", 
        "longitude", 
        "total cells/ml",
        "sample_date", 
        "satellite_date", 
        "temp_c", 
        "sensor", 
        "blue", "green", "red", "red_edge", "NIR", "SWIR", 
        "ndci", "ndvi", "ndwi", "ndti"
    ]
    test_processed = ices.map(extract_satellite_data)
    export = test_processed.select(final_cols)
    task = ee.batch.Export.table.toDrive(
        collection=export,
        description="test_satellite_matchup",
        fileFormat="CSV",
    )  
    task.start()
