import ee
import geemap
import pandas as pd
import numpy as np

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
    lat = coords.get(0)
    lon = coords.get(1)

    # create date window
    start = sample_date.advance(-5, "day")
    end = sample_date.advance(5, "day")

    # create a collection of potential s2 images based on location, dates, and cloud cover
    s2_collection = s2.filterBounds(geom) \
                      .filterDate(start, end) \
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25)) \
                      .sort("CLOUDY_PIXEL_PERCENTAGE")
    
    # create a collection of potential l8 images based on location, dates, and cloud cover
    l8_collection = l8.filterBounds(geom) \
                      .filterDate(start, end) \
                      .filter(ee.Filter.lt("CLOUD_COVER", 25)) \
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
        "satellite_date": "NONE", "temp_c": -999, 
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
        "sensor": sensor_type
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

if __name__ == "__main__":

    # initialize the library
    ee.Initialize(project="ml4cc-final-project")

    # get caml dataset (actual cyanobacteria samples)
    caml = ee.FeatureCollection("projects/sat-io/open-datasets/HAB-DETECTION/CAML_cyanobacteria_abundance_20211229_R1")

    # get satellite datasets
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")

    # process data
    old_cols = ["latitude", "longitude", "abun", "sample_date", "satellite_date", "temp_c", "sensor", "blue", "green", "red", "red_edge", "NIR", "SWIR", "ndci", "ndvi", "ndwi", "ndti"]
    new_cols = ["latitude", "longitude", "cyanobacteria_abundance", "sample_date", "satellite_date", "temp_c", "sensor", "blue", "green", "red", "red_edge", "NIR", "SWIR", "ndci", "ndvi", "ndwi", "ndti"]

    processed_data = caml.filter(ee.Filter.gte("date", 20130101)).map(extract_satellite_data)

    # export to google drive
    export = processed_data.select(old_cols, new_cols)
    task = ee.batch.Export.table.toDrive(
        collection=export,
        description="caml_satellite_matchup",
        fileFormat="CSV",
    )  
    task.start()