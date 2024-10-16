import osmnx as ox
import os
import geopandas as gpd
from tqdm.auto import tqdm
import numpy as np
from shapely.geometry import Point, LineString
import networkx as nx
from pyogrio import read_dataframe
from sklearn.neighbors import BallTree
import shutil
import logging

import warnings
warnings.filterwarnings('ignore')

from itertools import combinations

def explode(gdf):
    exploded_lines = {'geometry': [], 'highway': [], 'maxspeed': []}

    # Iterate over each row in the GeoDataFrame
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        # Get the LineString geometry and attributes
        geometry = row['geometry']
        attribute_1 = row['highway']
        attribute_2 = row['maxspeed']

        # Explode the LineString into separate segments
        segments = list(geometry.coords)
        for i in range(len(segments) - 1):
            segment = LineString([segments[i], segments[i + 1]])
            exploded_lines["geometry"].append(segment)
            exploded_lines['highway'].append(attribute_1)
            exploded_lines['maxspeed'].append(attribute_2)

    # Create a new GeoDataFrame from the exploded LineStrings
    exploded_gdf = gpd.GeoDataFrame(exploded_lines)
    exploded_gdf.crs = 'EPSG:4326'
    return exploded_gdf


def getEndPointsFromLine(linestring_obj):
    end1Coord = linestring_obj.coords[0]
    end2Coord = linestring_obj.coords[-1]
    return Point(*end1Coord), Point(*end2Coord)


def assignEndpoints(df_links, df_nodes):
    """
    Given a DataFrame with LineStrings and a DataFrame with Points, updates the 'end1' and 'end2' columns of the DataFrame with the indices of the closest Points, and optionally updates the geometry of the LineStrings. Returns the updated DataFrame.
    """
    # Create a BallTree index for the points in df_nodes
    tree = BallTree(df_nodes['geometry'].apply(lambda x: (x.x, x.y)).tolist(), leaf_size=2)

    def getNearestPoints(line):
        # Get the endpoints of the line
        end1, end2 = getEndPointsFromLine(line)

        # Find the nearest points in df_nodes using the BallTree index
        dist1, ind1 = tree.query([[end1.x, end1.y]], k=1)
        dist2, ind2 = tree.query([[end2.x, end2.y]], k=1)

        return ind1[0][0], ind2[0][0]

    tqdm.pandas()
    idClosestPointsArray = np.vstack(df_links['geometry'].progress_apply(getNearestPoints))
    df_links['end1'] = idClosestPointsArray[:, 0]
    df_links['end2'] = idClosestPointsArray[:, 1]
    return df_links.astype({'end1': int, 'end2': int})


def simplification_A(input_folder):
    logging.info("SIMPLIFICATION A PHASE")
    input_folder = os.path.join(os.path.dirname(input_folder), "raw_files")
    files = set([file[:-14] for file in os.listdir(os.path.join(input_folder)) if
                 os.path.isfile(os.path.join(os.path.join(input_folder), file))])

    for i in files:
        logging.info("simplify: " + i)
        gpd_edges = read_dataframe(os.path.join(input_folder, i + '_edges.geojson'),
                                   columns=["highway", "maxspeed", "geometry"])

        logging.debug("exploding edges")
        gpd_edges_exploded = explode(gpd_edges)
        del gpd_edges

        gpd_edges_exploded['osmid'] = gpd_edges_exploded.index
        gpd_edges_exploded['key'] = gpd_edges_exploded["osmid"]

        gpd_nodes = read_dataframe(os.path.join(input_folder, i + '_nodes.geojson'))
        gpd_nodes = gpd_nodes.loc[:, ["geometry"]]
        gpd_nodes['x'] = gpd_nodes['geometry'].x
        gpd_nodes['y'] = gpd_nodes['geometry'].y

        logging.debug("assigning endpoints")
        gpd_edges_exploded = assignEndpoints(df_links=gpd_edges_exploded, df_nodes=gpd_nodes)

        logging.info(gpd_edges_exploded['highway'].value_counts())

        gpd_nodes['osmid'] = gpd_nodes.index
        gpd_nodes_test = gpd_nodes.set_index('osmid')
        gpd_edges_test = gpd_edges_exploded.set_index(['end1', 'end2', 'key'])
        del gpd_nodes
        del gpd_edges_exploded
        G_test = ox.utils_graph.graph_from_gdfs(gpd_nodes_test, gpd_edges_test, graph_attrs=None)
        del gpd_nodes_test
        del gpd_edges_test
        logging.debug("Graph created")

        G_proj_test = ox.project_graph(G_test)
        logging.debug("Graph projected")
        del G_test

        G_simp = ox.simplification.consolidate_intersections(G_proj_test, tolerance=40, rebuild_graph=True,
                                                             dead_ends=False, reconnect_edges=True)
        del G_proj_test

        for u, v, k, data in G_simp.edges(keys=True, data=True):
            for attr, value in data.items():
                if isinstance(value, list):
                    data[attr] = tuple(value)

        G_simp2 = ox.simplification.simplify_graph(G_simp, strict=True, remove_rings=True, track_merged=False)
        G_simp2 = ox.project_graph(G_simp2, to_crs='EPSG:4326')

        # Create a new graph to store the filtered edges
        G_simp3 = nx.MultiDiGraph()
        G_simp3.graph["crs"] = 'EPSG:4326'

        # Iterate over the edges of the original graph
        for u, v, key, attr in G_simp2.edges(keys=True, data=True):
            # Check if the edge (u, v) exists in the new graph
            if not (G_simp3.has_edge(u, v) or G_simp3.has_edge(v, u)):
                # If the edge does not exist, add it to the new graph
                G_simp3.add_edge(u, v, key=key, **attr)

        # Copy all nodes from the original graph to the new graph
        G_simp3.add_nodes_from(G_simp2.nodes(data=True))

        for u, v, k, data in G_simp3.edges(keys=True, data=True):
            for attr, value in data.items():
                if isinstance(value, list):
                    data[attr] = tuple(value)
        G_simp4 = ox.simplification.simplify_graph(G_simp3, strict=True, remove_rings=True, track_merged=False)

        logging.debug("Saving and moving processed file to 'Done' folder")
        output_folder = os.path.join(os.path.dirname(input_folder), "simp_A")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(os.path.join(input_folder, "Done")):
            os.makedirs(os.path.join(input_folder, "Done"))

        gdf = ox.save_graph_geopackage(G_simp4, filepath=os.path.join(output_folder, i + '.gpkg'), directed=False)
        shutil.move(os.path.join(input_folder, i + '_edges.geojson'),
                    os.path.join(input_folder, "Done", i + '_edges.geojson'))
        shutil.move(os.path.join(input_folder, i + '_nodes.geojson'),
                    os.path.join(input_folder, "Done", i + '_nodes.geojson'))