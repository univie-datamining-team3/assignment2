
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

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


def get_color_encoding(color_coding):
    """
    Helper function to generate color map and color patches.
    :return colors: mapping of seaborn rgb colors to each points
    :return color_patches: color patches for each rgb color, will be displayed
                           in the legend of the plot.
    """
    unique_labels = list(set(list(color_coding)))
    nr_of_labels = len(unique_labels)
    color_palette = sns.color_palette("hls", nr_of_labels)
    color_mapping = dict()
    for label_i, color in zip(unique_labels, color_palette):
        color_mapping[label_i] = color

    color_coding_copy = list(color_coding)
    colors = list(map(lambda x: color_mapping[x], color_coding_copy))
    color_patches = []
    for label, label_color in color_mapping.items():
        color_patches.append(mpl.patches.Patch(color=label_color, label=str(label)))

    return colors, color_patches

def _find_contiguous_colors(colors):
    """
    Helper function that finds the continuous segments of colors
    and returns those segments
    Code adapted from:
    http://abhay.harpale.net/blog/python/how-to-plot-multicolored-lines-in-matplotlib/
    """
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg) # the final one
    return segs

def plot_multicolored_lines(x,y,colors, ax=None):
    """
    Plotting function that allows easy plotting of multicolored continuous
    1D Time Series. colors could be the cluster labels.
    Code adapted from:
    http://abhay.harpale.net/blog/python/how-to-plot-multicolored-lines-in-matplotlib/
    """
    segments = _find_contiguous_colors(colors)
    #plt.figure(figsize=(15, 5))
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    start= 0
    for seg in segments:
        end = start + len(seg)
        l, = ax.plot(x[start:end],y[start:end],lw=1,c=seg[0])
        start = end


def plot_timeseries_clustering(x_time, y, labels, ax=None):
    """
    Plotting function to plot colored timeseries, where each segment, corresponds
    to a cluster label.
    """
    colors, color_patches = get_color_encoding(labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              handles=color_patches)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70, horizontalalignment='right')
    ax.xaxis.set_major_locator(mdates.MinuteLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d:%H:%M:%S'))
    plot_multicolored_lines(x_time,y, colors, ax=ax)

def _get_cluster_labels(labels, factor=600):
    """
    Each label is repeated factor=600 times. E.g. labels is a list
    with [1,2] of length 2, then this method returns a list of length
    1200 and the list is [1,1,1...,2,2,2]
    """
    # Make sure labels is iterable
    copy = list(labels)
    labels_multiplied = []
    for label_list in ([i]*factor for i in copy):
        labels_multiplied += label_list
    return labels_multiplied

def get_plot_timeseries_clustering_variables(distance_matrix, trips, trip_id, sensor_type="acceleration", x_column="time", y_column="total"):
    """
    This function helps to get the prepared variables as input
    for plot_timeseries_clustering(x_time, y, labels, ax=None).

    Parameters
    ----------
    distance_matrix: pandas.DataFrame of calculated distances
        from the trip segments and addionally these
        ["mode","notes","scripted","token", "trip_id"] columns
    trips: all recorded trips as a list.
    trip_id: int, specifies which trip will be plotted.
    sensor_type: string, default="acceleration",
        specifies which sensor type should be plotted
    x_column: string, default="time",
        specifies x_axis for the plot, because we have timeseries data this
        should be a column of datetime values
    y_column: string, default="total",
        specifies y_axis for the plot


    Returns
    -------
    x_column values for one trip
    y_column values for one trip
    cluster labels for one trip
    """
    small_df_trip = distance_matrix[distance_matrix.trip_id == trip_id]
    helper = trips[trip_id]["sensor"]
    helper = helper[helper.sensor == sensor_type]
    # Important, because indices are not unique
    sensor_data_trip_i = helper.reset_index(drop=True)

    cluster_labels = _get_cluster_labels(small_df_trip["cluster_labels"])
    diff = sensor_data_trip_i.shape[0] - len(cluster_labels)
    rows_to_be_dropped = sensor_data_trip_i.tail(diff).index
    sensor_data_trip_i = sensor_data_trip_i.drop(rows_to_be_dropped)
    sensor_data_trip_i["cluster_labels"]= cluster_labels

    return (list(sensor_data_trip_i[x_column]), list(sensor_data_trip_i[y_column]),
            cluster_labels)
