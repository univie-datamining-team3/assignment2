
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
