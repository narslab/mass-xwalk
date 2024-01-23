# mass-xwalk

This repository contains the scripts required to perform the post-processing on the crosswalk polygons detected in the project "Artificial intelligence framework for midblock crosswalk detection across Massachusetts." The post-processing involves the steps of filtering out false positive detections and categorizing detected crosswalks as *intersection*, *midblock* and *driveway*.

It contains three scripts in the `bin` folder.

1. `post_processing_results.ipynb`
This is a Jupyter notebook that demonstrates the functionalities of the post-processing framework. Importantly, it also contains the functions (`superimpose_polygon_on_tile`) for visualizing the polygons, buffers and road network superimposed over the tile images.

2. `post_processing_results_dissolve_split.py`
This script performs the entire post-processing routine using the "Dissolve Split" road network.

3. `post_processing_results_road_arc.py`
This script performs the entire post-processing routine using the "Road Arc" network. Final counts were reported using the results of this script.


## Inputs
The following inputs and their respective relative paths are required to run the code:

### Shapefiles
`post_processing/remaymonthlyreportformassxwalkproject/Image_Index_2019/COQ2019INDEX_POLY.shp`
`post_processing/remaymonthlyreportformassxwalkproject/Image_Index_2021/COQ2021INDEX_POLY.shp`

### Detected crosswalks
`post_processing/Results_2019_2021.gdb/`
`post_processing/MassDOT_Roads_GDB/MassDOT_Roads.gdb/`

### Cleaned Intersection data
`post_processing/Cleaned_Intersection_Data/Cleaned_Intersection_Data.gdb/`

### Road network
`post_processing/MassDOT_Roads_GDB/MassDOT_Roads.gdb/` (original network)
`Dissolve_Split/Dissolve_Split/` (dissolve split network)
