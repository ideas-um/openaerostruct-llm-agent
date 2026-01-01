###### Import functions are here, DO NOT EDIT ##########
import warnings
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import openmdao.api as om
# import OpenAeroStruct modules
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
import niceplots  # Optional but recommended
import time

""" CORE FUNCTIONS ARE DEFINED HERE, DO NOT EDIT """
def plot_mesh(mesh):
    """ this function plots to plot the mesh """
    mesh_x = mesh[:, :, 0]
    mesh_y = mesh[:, :, 1]
    plt.figure(figsize=(6, 3))
    color = 'k'
    for i in range(mesh_x.shape[0]):
        plt.plot(mesh_y[i, :], mesh_x[i, :], color, lw=1)
        # plt.plot(-mesh_y[i, :], mesh_x[i, :], color, lw=1)   # plots the other side of symmetric wing
    for j in range(mesh_x.shape[1]):
        plt.plot(mesh_y[:, j], mesh_x[:, j], color, lw=1)
        # plt.plot(-mesh_y[:, j], mesh_x[:, j], color, lw=1)   # plots the other side of symmetric wing
    plt.axis('equal')
    plt.xlabel('span(m)')
    plt.ylabel('chord(m)')
    plt.savefig('./figures/mesh.pdf', bbox_inches='tight')

"""Part 1: PUT THE BASELINE MESH OF THE WING HERE"""



"""Part 2:  DO THE GEOMETRY SETUP HERE"""



"""Part 3: PUT THE OPTIMIZER HERE """

