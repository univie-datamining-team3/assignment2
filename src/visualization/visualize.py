
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.patches as mpatches

import gmplot
import os

def setup_directory(dir_name):
    """Setup directory in case it does not exist
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print("Created Directory: {}".format(dir_name) )
        except:
            print("Could not create directory: {}".format(dir_name))

def get_vis_dir():
    """ Returns the data dir relative from this file
    """
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    vis_dir = os.path.join(project_dir,"reports")
    return vis_dir


def plot_track(track, file_name):
    '''
    Plots a track onto google maps. produces a <file_name>.html file in
    '''
    gmas = gmplot.GoogleMapPlotter(track["latitude"].iloc[0],track["longitude"].iloc[0], 18)
    gmas.plot(track["latitude"].iloc[1:],track["longitude"].iloc[1:])

    if not file_name.endswith(".html"):
        file_name = file_name + '.html'
    map_dir = os.path.join(get_vis_dir(),"maps")
    setup_directory(map_dir)
    gmas.draw(os.path.join(map_dir, file_name))

def plot_gps_heatmap(tracks, file_name):
    """
    Plots all tracks onto google map and visualizes them as heatmap
    """
    latitudes = pd.DataFrame()
    longitudes = pd.DataFrame()

    for trip_i in tracks:
        location_i = trip_i["location"]
        if location_i is not None and not location_i.empty:

            latitude_i = location_i['latitude']
            longitude_i = location_i['longitude']
            latitudes = pd.concat([latitudes, latitude_i])
            longitudes = pd.concat([longitudes, longitude_i])

    gmap = gmplot.GoogleMapPlotter.from_geocode("Vienna")
    gmap.heatmap(list(latitudes[0]), list(longitudes[0]))

    map_dir = os.path.join(get_vis_dir(),"maps")
    setup_directory(map_dir)
    if not file_name.endswith(".html"):
        file_name = file_name + '.html'
    gmap.draw(os.path.join(map_dir, file_name))



def plot_acceleration_sensor(acceleration_for_one_trip: pd.DataFrame):
    figsize=(12, 4)
    acceleration_for_one_trip["x"].plot(figsize=figsize);
    plt.ylabel("x")
    plt.show();

    acceleration_for_one_trip["y"].plot(figsize=figsize);
    plt.ylabel("y")
    plt.show();

    acceleration_for_one_trip["z"].plot(figsize=figsize);
    plt.ylabel("z")
    plt.show();

    acceleration_for_one_trip["total"].plot(figsize=figsize);
    plt.ylabel("total")
    plt.show();
