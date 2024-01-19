from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
import time

import geopandas as gpd
import pandas as pd
import os
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
import fiona

import matplotlib.pyplot as plt
import seaborn as sns

import rtree

from PIL import Image

import brewer2mpl

import requests
import zipfile

from shapely.geometry import Point, Polygon, MultiPolygon, MultiLineString
from shapely.ops import unary_union
from shapely.wkt import loads

import rasterio as rio
import rasterio.mask
from rasterio.plot import show
from shapely.geometry import mapping
from rasterio.transform import Affine



def get_quantiles(res):
    quantiles = res['Shape_Area'].quantile([0.25, 0.5, 0.75])
    source = ' '.join(word.capitalize() for word in res['source'][0].split('_'))
    stats = [[quantiles.keys()[0], quantiles[quantiles.keys()[0]]],
     [quantiles.keys()[1], quantiles[quantiles.keys()[1]]],
     [quantiles.keys()[2], quantiles[quantiles.keys()[2]]],]
    table = tabulate(stats, headers=["Quantile", "Value"], tablefmt="fancy_grid")
    print(f"Quantiles: {source} {str(res['year'][0])}")
    print(table)
    return(quantiles)

#### Previous implementation 
# def filter_based_on_area(crosswalks, area_threshold=0.75, area_threshold_adjustment=0):
#     crosswalks_transformed = crosswalks.copy()
#     # Calculate statistics
#     quantiles = get_quantiles(crosswalks_transformed)
#     # Calculate the area threshold directly from quantiles
#     area_threshold_value = quantiles[area_threshold]    
#     # Use np.where for efficient condition-based column creation
#     crosswalks_transformed['category'] = np.where(
#         crosswalks_transformed['geometry'].area >= (area_threshold_value - area_threshold_adjustment),
#         'tp',
#         'fp'
#     )
#     # Drop the existing index column if needed
#     crosswalks.reset_index(drop=True, inplace=True)    
#     return crosswalks_transformed

def filter_based_on_area(crosswalks, area_threshold=10):
    crosswalks_transformed = crosswalks.copy()   
    # Use np.where for efficient condition-based column creation
    crosswalks_transformed['category'] = np.where(
        crosswalks_transformed['geometry'].area >= area_threshold,
        'tp',
        'fp'
    )
    # Drop the existing index column if needed
    crosswalks.reset_index(drop=True, inplace=True)    
    return crosswalks_transformed

def get_true_positives(gdf):
    # Extract 'tilename' from 'MERGE_SRC'
    gdf['tilename'] = gdf['MERGE_SRC'].str.extract(r'Res_(.*?)_Res')

    # Create an 'object_id' column using the index of filtered_gdf
    gdf['object_id'] = gdf.index + 1
    
    # Filter rows in gdf where the 'category' is not 'fp'
    filtered_gdf = gdf[gdf['category'] == 'tp'].copy()

    # Create a 'type' column with a default value
    filtered_gdf['type'] = 'other'

    # Map the entries from the filtered gdf back to the original gdf using the row index as the key
    gdf['type'] = gdf.index.map(filtered_gdf.set_index(filtered_gdf.index)['type'])

    # Reorder columns with 'object_id' as the first column
    gdf = gdf[['object_id'] + [col for col in gdf.columns if col != 'object_id']]
    
    return gdf


def set_polygon_crs(polygon_df, road_df, intersection_df):
    if polygon_df.crs != road_df.crs:
        road_df_transformed = road_df.to_crs(polygon_df.crs)
    else:
        road_df_transformed = road_df.crs
        
    if polygon_df.crs != intersection_df.crs:
        intersection_df_transformed = intersection_df.to_crs(polygon_df.crs)
    else:
        intersection_df_transformed = intersection_df.crs        
        
    return(road_df_transformed, intersection_df_transformed)
    

def find_intersecting_roads(polygon_df, road_df, polygon_buffer_size=5):
    # Perform a spatial join based on intersection
    intersecting_roads_df = gpd.sjoin(polygon_df[['geometry']].apply(lambda geom: geom.buffer(polygon_buffer_size)), road_df[['geometry']], how='inner', op='intersects')

    # Group the intersecting road segments and add them to lists
    grouped_roads = intersecting_roads_df.groupby(intersecting_roads_df.index)['index_right'].apply(list)

    # Merge the grouped road segments back into 'fdf'
    polygon_df['intersecting_roads'] = polygon_df.index.map(grouped_roads)
    
    return polygon_df

def find_intersection_points(polygon_df, road_df, intersection_pt_df, intersection_pt_buffer_size=30, polygon_buffer_size=10):
    buffer_params = {'source':polygon_df['source'].iloc[0],
                     'year':polygon_df['year'].iloc[0],
                     'intersection_pt_buffer_size':intersection_pt_buffer_size,
                     'polygon_buffer_size':polygon_buffer_size}
    
    other_type = polygon_df[polygon_df['type']=='other'].copy()
    
    # Spatially join the two GeoDataFrames based on the 'geometry' column
    intersections = gpd.sjoin(other_type[['geometry']], intersection_pt_df[['geometry']].apply(lambda pt: pt.buffer(intersection_pt_buffer_size)), how='inner', op='intersects')

    # Group the intersecting road segments and add them to lists
    grouped_intersections = intersections.groupby(intersections.index)['index_right'].apply(list)

    # Merge the grouped road segments back into the filtered dataframe
    other_type['intersection_points'] = other_type.index.map(grouped_intersections)

    # Update the 'type' column with 'intersection' for intersecting rows
    other_type.loc[grouped_intersections.index, 'type'] = 'intersection'
    other_type = find_intersecting_roads(other_type, road_df, polygon_buffer_size)

    polygon_df['type'] = polygon_df.index.map(other_type.set_index(other_type.index)['type'])
    polygon_df['intersection_points'] = polygon_df.index.map(other_type.set_index(other_type.index)['intersection_points'])
    polygon_df['intersecting_roads'] = polygon_df.index.map(other_type.set_index(other_type.index)['intersecting_roads'])

    polygon_df = polygon_df.drop('index_right', axis=1, errors='ignore')
    # print(polygon_df.head())
    
    return(polygon_df, buffer_params)

def midblock_test(polygon_df, road_df, threshold=5):
    other_rows = polygon_df[polygon_df['type'] == 'other'].copy()
    
    if not other_rows.empty:
        other_rows = find_intersecting_roads(other_rows, road_df, 0)
        intersecting_rds = other_rows[~other_rows['intersecting_roads'].isna()]

        intersecting_rds['type'] = intersecting_rds.apply(
            lambda row: 'other' if pd.isna(row['intersecting_roads']).any() else row['type'], axis=1)

        intersecting_rds['road_geometries'] = intersecting_rds['intersecting_roads'].apply(
            lambda idx_list: [road_df.iloc[road_segment_id]['geometry'] for road_segment_id in idx_list])

        # Define a lambda function to perform the overlay operation
        overlay_lambda = lambda road_geometries, polygon_geometry: gpd.overlay(
            gpd.GeoDataFrame(geometry=road_geometries),
            gpd.GeoDataFrame(geometry=[polygon_geometry]),
            how='difference', keep_geom_type=True
        ).explode(index_parts=False).reset_index(drop=True)['geometry'].tolist()

        intersecting_rds['results'] = intersecting_rds.apply(
            lambda row: overlay_lambda(row['road_geometries'], row['geometry']), axis=1)

        intersecting_rds['type'] = intersecting_rds['results'].apply(
            lambda results: 'midblock' if all(segment.length > threshold for segment in results) else 'other'
        )

        # Update the 'type' and 'non_intersecting_segments' columns in polygon_df
        polygon_df.loc[intersecting_rds.index, 'type'] = intersecting_rds['type']
        polygon_df.loc[intersecting_rds.index, 'non_intersecting_segments'] = intersecting_rds['results']

    return polygon_df
    
# Helper function to create the buffer based on road_type
def create_buffered_geometry(road_df, idx_list):
    buffer_size = 10 if all(road_df.iloc[road_segment_id]['RDTYPE'] <= 4 for road_segment_id in idx_list) else 8
    return [road_df.iloc[road_segment_id]['geometry'].buffer(buffer_size) for road_segment_id in idx_list]    

def assign_type(row):
    if row['type'] == 'other':
        return 'driveway' if any(row['geometry'].intersects(buffer) for buffer in row['road_geometry_buffer']) else 'parking'
    else:
        return row['type']

def driveway_test(polygon_df, road_df):
    other_rows = polygon_df[polygon_df['type'] == 'other']
    
    if not other_rows.empty:
        other_rows = find_intersecting_roads(other_rows, road_df, 20)
        intersecting_rds = other_rows[~other_rows['intersecting_roads'].isna()]
        
        intersecting_rds['type'] = intersecting_rds.apply(
            lambda row: 'other' if pd.isna(row['intersecting_roads']).any() else row['type'], axis=1)
        
        intersecting_rds['road_geometry_buffer'] = intersecting_rds['intersecting_roads'].apply(lambda row: create_buffered_geometry(road_df, row))
        intersecting_rds['type'] = intersecting_rds.apply(assign_type, axis=1)        
        
        polygon_df.loc[intersecting_rds.index, 'road_geometry_buffer'] = intersecting_rds['road_geometry_buffer']
        polygon_df.loc[intersecting_rds.index, 'type'] = intersecting_rds['type']
    
    return polygon_df

def print_counts(df):
    print(f"{df.iloc[0]['source']} {df.iloc[0]['year']}")
    print(f"Intersection: {len(df[df['type']=='intersection'])}")
    print(f"Midblock: {len(df[df['type']=='midblock'])}")
    print(f"Driveway: {len(df[df['type']=='driveway'])}")
    print(f"Parking: {len(df[df['type']=='parking'])}")
    print(f"Other: {len(df[df['type']=='other'])}")
    print(f"False positive: {len(df[df['category']=='fp'])}")
    print(f"{len(df)}\n")
    
def calculate_num_crosswalks_parallel(args):
    row, roads_geometry = args
    polygon_gdf = gpd.GeoDataFrame(geometry=[row['geometry']], crs=roads_geometry.crs)
    
    # Replace sjoin with overlay
    intersection_gdf = gpd.overlay(polygon_gdf, roads_geometry, how='intersection')
    
    return len(intersection_gdf)
    
def count_all_intersections_parallel(df, roads, num_processes=4):
    df_copy = df.copy()
    df_split_intersections_only = df_copy[(df_copy['type'] == 'intersection') & (~df_copy['intersecting_roads'].isna())]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        args = ((row, roads) for _, row in df_split_intersections_only.iterrows())
        results = list(executor.map(calculate_num_crosswalks_parallel, args))

    df_copy.loc[df_split_intersections_only.index, 'num_crosswalks'] = results
    return df_copy    

def calculate_num_crosswalks(row, roads):
    polygon_gdf = gpd.GeoDataFrame(geometry=[row['geometry']])
    roads_gdf = gpd.GeoDataFrame(geometry=roads['geometry'][row['intersecting_roads']])
    intersection_gdf = gpd.sjoin(polygon_gdf, roads_gdf, how='inner', predicate='intersects')
    return len(intersection_gdf)

def count_all_intersections(df, roads):
    df_copy = df.copy()
    df_split_intersections_only = df_copy[(df_copy['type']=='intersection') & (~df_copy['intersecting_roads'].isna())]
    df_split_intersections_only['num_crosswalks'] = df_split_intersections_only.apply(lambda row: calculate_num_crosswalks(row, roads), axis=1)
    df_copy.loc[df_split_intersections_only.index, 'num_crosswalks'] = df_split_intersections_only['num_crosswalks']
    return df_copy

def map_gridcode_to_class(gridcode):
    if gridcode == 1:
        return 'continental'
    elif gridcode == 2:
        return 'parallel'
    elif gridcode == 3:
        return 'solid'
    else:
        return 'unknown'  # You can handle other values if any

def main(): 
    shape_file_path_2019 = 'post_processing/remaymonthlyreportformassxwalkproject/Image_Index_2019/COQ2019INDEX_POLY.shp'
    shape_file_path_2021 = 'post_processing/remaymonthlyreportformassxwalkproject/Image_Index_2021/COQ2021INDEX_POLY.shp'
    results_gdb = 'post_processing/Results_2019_2021.gdb/'
    roads_gdb = 'post_processing/MassDOT_Roads_GDB/MassDOT_Roads.gdb/'

    clean_intersection_data = 'post_processing/Cleaned_Intersection_Data/Cleaned_Intersection_Data.gdb/'

    results_path = os.path.join(results_gdb)
    roads_path = os.path.join(roads_gdb)
    clean_intersections_path = os.path.join(clean_intersection_data)

    start = time.time()
    at_grade_intersections = gpd.read_file(clean_intersections_path, driver='FileGDB', layer=2)
    elapsed = time.time() - start
    print(f"Loaded at-grade intersection in: {timedelta(seconds=elapsed)} h/m/s")

    # 2019 & 2021 shape files
    shape_file_2019 = gpd.read_file(shape_file_path_2019)
    shape_file_2021 = gpd.read_file(shape_file_path_2021)

    # 2019 results
    start = time.time()
    western_mass_2019 = gpd.read_file(results_path, driver='FileGDB', layer=2)
    eastern_mass_2019 = gpd.read_file(results_path, driver='FileGDB', layer=3)
    elapsed = time.time() - start
    print(f"Loaded 2019 results in: {timedelta(seconds=elapsed)} h/m/s")

    # 2021 results
    start = time.time()
    western_mass_2021 = gpd.read_file(results_path, driver='FileGDB', layer=0)
    eastern_mass_2021 = gpd.read_file(results_path, driver='FileGDB', layer=1)
    elapsed = time.time() - start
    print(f"Loaded 2021 results in: {timedelta(seconds=elapsed)} h/m/s")

    # Road network
    start = time.time()
    roads_arc = gpd.read_file(roads_path, driver='FileGDB', layer=2)
    # Filter out private roads and those unaccepted by city or town
    roads_arc_filtered = roads_arc[~roads_arc['JURISDICTN'].isin(['0', 'H'])].reset_index(drop=True)    
#     dissolve_split = gpd.read_file('Dissolve_Split/Dissolve_Split/')
    elapsed = time.time() - start
    print(f"Loaded road network in {timedelta(seconds=elapsed)} h/m/s")

    # Add source (Eastern/Western) and year (2019/2021)
    western_mass_2019['source'] = 'western_mass'
    western_mass_2019['year'] = 2019
    eastern_mass_2019['source'] = 'eastern_mass'
    eastern_mass_2019['year'] = 2019
    western_mass_2021['source'] = 'western_mass'
    western_mass_2021['year'] = 2021
    eastern_mass_2021['source'] = 'eastern_mass'
    eastern_mass_2021['year'] = 2021
   
    wm21 = filter_based_on_area(western_mass_2021, area_threshold=20)
    em21 = filter_based_on_area(eastern_mass_2021, area_threshold=20)
    wm19 = filter_based_on_area(western_mass_2019, area_threshold=20)
    em19 = filter_based_on_area(eastern_mass_2019, area_threshold=20)
    
    wm21_filtered = get_true_positives(wm21)
    em21_filtered = get_true_positives(em21)
    wm19_filtered = get_true_positives(wm19)
    em19_filtered = get_true_positives(em19)
    
    roads_arc_transformed_west21, intersection_pts_transformed_west21 = set_polygon_crs(wm21_filtered, roads_arc_filtered, at_grade_intersections)
    roads_arc_transformed_east21, intersection_pts_transformed_east21 = set_polygon_crs(em21_filtered, roads_arc_filtered, at_grade_intersections)
    roads_arc_transformed_west19, intersection_pts_transformed_west19 = set_polygon_crs(wm19_filtered, roads_arc_filtered, at_grade_intersections)
    roads_arc_transformed_east19, intersection_pts_transformed_east19 = set_polygon_crs(em19_filtered, roads_arc_filtered, at_grade_intersections)
    
    start = time.time()
    wm21_intersection, intersection_pt_params_west21 = find_intersection_points(wm21_filtered, roads_arc_transformed_west21, intersection_pts_transformed_west21)
    em21_intersection, intersection_pt_params_east21 = find_intersection_points(em21_filtered, roads_arc_transformed_east21, intersection_pts_transformed_east21)
    wm19_intersection, intersection_pt_params_west19 = find_intersection_points(wm19_filtered, roads_arc_transformed_west19, intersection_pts_transformed_west19)
    em19_intersection, intersection_pt_params_east19 = find_intersection_points(em19_filtered, roads_arc_transformed_east19, intersection_pts_transformed_east19)
    elapsed = time.time() - start
    print(f"Categorized intersections in {timedelta(seconds=elapsed)} h/m/s")
    
    start = time.time()
    wm21_intersection_midblock = midblock_test(wm21_intersection, roads_arc_transformed_west21, 8)
    em21_intersection_midblock = midblock_test(em21_intersection, roads_arc_transformed_east21, 8)
    wm19_intersection_midblock = midblock_test(wm19_intersection, roads_arc_transformed_west19, 8)
    em19_intersection_midblock = midblock_test(em19_intersection, roads_arc_transformed_east19, 8)
    elapsed = time.time() - start
    print(f"Categorized midblocks in {timedelta(seconds=elapsed)} h/m/s")
    
    start = time.time()
    wm21_intersection_midblock_driveway = driveway_test(wm21_intersection_midblock, roads_arc_transformed_west21)
    em21_intersection_midblock_driveway = driveway_test(em21_intersection_midblock, roads_arc_transformed_east21)
    wm19_intersection_midblock_driveway = driveway_test(wm19_intersection_midblock, roads_arc_transformed_west19)
    em19_intersection_midblock_driveway = driveway_test(em19_intersection_midblock, roads_arc_transformed_east19)
    elapsed = time.time() - start
    print(f"Categorized driveways in {timedelta(seconds=elapsed)} h/m/s")
    
    complete_set = [em19_intersection_midblock_driveway, wm21_intersection_midblock_driveway, wm19_intersection_midblock_driveway, em21_intersection_midblock_driveway]
    
    for s in complete_set:
        print_counts(s)
                
    start = time.time()
    wm21_count_final = count_all_intersections(wm21_intersection_midblock_driveway, roads_arc_transformed_west21)
    wm19_count_final = count_all_intersections(wm19_intersection_midblock_driveway, roads_arc_transformed_west19)
    em21_count_final = count_all_intersections(em21_intersection_midblock_driveway, roads_arc_transformed_east21)
    em19_count_final = count_all_intersections(em19_intersection_midblock_driveway, roads_arc_transformed_east19)
    elapsed = time.time() - start
    print(f"Counted intersecting segments (intersection only) in {timedelta(seconds=elapsed)} h/m/s")
    
    wm21_count_final['class'] = wm21_count_final['gridcode'].apply(map_gridcode_to_class)
    wm19_count_final['class'] = wm19_count_final['gridcode'].apply(map_gridcode_to_class)
    em21_count_final['class'] = em21_count_final['gridcode'].apply(map_gridcode_to_class)
    em19_count_final['class'] = em19_count_final['gridcode'].apply(map_gridcode_to_class)
    
    # Drop rows where 'type' is 'other' or 'parking', or 'category' is 'fp'
#     wm21_count_final_filtered = wm21_count_final[~((wm21_count_final['type'] == 'other') | (wm21_count_final['type'] == 'parking') | (wm21_count_final['category'] == 'fp'))]
#     wm19_count_final_filtered = wm19_count_final[~((wm19_count_final['type'] == 'other') | (wm19_count_final['type'] == 'parking') | (wm19_count_final['category'] == 'fp'))]
#     em21_count_final_filtered = em21_count_final[~((em21_count_final['type'] == 'other') | (em21_count_final['type'] == 'parking') | (em21_count_final['category'] == 'fp'))]
#     em19_count_final_filtered = em19_count_final[~((em19_count_final['type'] == 'other') | (em19_count_final['type'] == 'parking') | (em19_count_final['category'] == 'fp'))]
    
    #wm21_count_final.to_csv(f"{wm21_count_final.iloc[0]['source']}_{wm21_count_final.iloc[0]['year']}.gdb", index=False)
    #wm19_count_final.to_csv(f"{wm19_count_final.iloc[0]['source']}_{wm19_count_final.iloc[0]['year']}.gdb", index=False)
    #em21_count_final.to_csv(f"{em21_count_final.iloc[0]['source']}_{em21_count_final.iloc[0]['year']}.gdb", index=False)
    #em19_count_final.to_csv(f"{em19_count_final.iloc[0]['source']}_{em19_count_final.iloc[0]['year']}.gdb", index=False)
    
    today_date = datetime.now().strftime('%m_%d_%Y')
    output_dir = f'results/{today_date}/roads_arc/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Save GeoDataFrame to csv
    wm21_filename = f"{wm21_count_final.iloc[0]['source']}_{wm21_count_final.iloc[0]['year']}_roads_arc.csv"
    wm19_filename = f"{wm19_count_final.iloc[0]['source']}_{wm19_count_final.iloc[0]['year']}_roads_arc.csv"
    em21_filename = f"{em21_count_final.iloc[0]['source']}_{em21_count_final.iloc[0]['year']}_roads_arc.csv"
    em19_filename = f"{em19_count_final.iloc[0]['source']}_{em19_count_final.iloc[0]['year']}_roads_arc.csv"
    
    wm21_filepath = output_dir + wm21_filename
    wm19_filepath = output_dir + wm19_filename
    em21_filepath = output_dir + em21_filename
    em19_filepath = output_dir + em19_filename
    
    print(f"{wm21_filepath}\n{wm19_filepath}\n{em21_filepath}\n{em19_filepath}")
    
    wm21_count_final.to_csv(wm21_filepath, index=False)
    wm19_count_final.to_csv(wm19_filepath, index=False)
    em21_count_final.to_csv(em21_filepath, index=False)
    em19_count_final.to_csv(em19_filepath, index=False)
    
    print(f"Saved complete results for {wm21_count_final.iloc[0]['source']},{wm21_count_final.iloc[0]['year']}.")    
    print(f"Saved complete results for {wm19_count_final.iloc[0]['source']},{wm19_count_final.iloc[0]['year']}.")    
    print(f"Saved complete results for {em21_count_final.iloc[0]['source']},{wm21_count_final.iloc[0]['year']}.")    
    print(f"Saved complete results for {em19_count_final.iloc[0]['source']},{em19_count_final.iloc[0]['year']}.")    
if __name__ == "__main__":
    main()