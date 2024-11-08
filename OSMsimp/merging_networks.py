import os
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from shapely.geometry import Point, LineString
from pyogrio import read_dataframe
from pyogrio import write_dataframe
from sklearn.neighbors import BallTree
import logging
from itertools import combinations
import warnings


def calculate_adaptive_threshold(merged_nodes):
    # Calculer les distances aux nœuds les plus proches
    distances = merged_nodes.geometry.apply(
        lambda geom: merged_nodes.distance(geom).nsmallest(2).iloc[-1]
    )
    # Calculer le 95ᵉ percentile
    threshold = distances.quantile(0.05)
    return threshold*1.3

def merge_geodataframes(geodataframes, distance_threshold=None):
    merged_nodes = gpd.GeoDataFrame()
    merged_edges = gpd.GeoDataFrame()

    # Merge all the nodes into a single GeoDataFrame
    for nodes, _ in geodataframes:
        nodes = nodes.to_crs(crs='4087')
        merged_nodes = pd.concat([merged_nodes, nodes], ignore_index=True)

    # Merge all the edges into a single GeoDataFrame
    for _, edges in geodataframes:
        edges = edges.to_crs(crs='4087')
        merged_edges = pd.concat([merged_edges, edges], ignore_index=True)

    # Create a new GeoDataFrame to store the connected edges
    connected_edges = gpd.GeoDataFrame(columns=merged_nodes.columns)

    # Iterate over combinations of merged nodes and find nearby nodes to create connected edges
    for (source_nodes, source_edges), (target_nodes, target_edges) in combinations(geodataframes, 2):
        # Project the source and target nodes to CRS 4087
        source_nodes = source_nodes.to_crs("EPSG:4087")
        target_nodes = target_nodes.to_crs("EPSG:4087")

        if distance_threshold == None:
            distance_threshold = calculate_adaptive_threshold(merged_nodes)
        # logging.info(f"Distance_treshold : {distance_threshold}")
        print(f"Distance_treshold : {distance_threshold}")
        for _, source_node in source_nodes.iterrows():
            # Transform the source node geometry to CRS 4087
            source_node_geometry = source_node.geometry  # .to_crs("EPSG:4087")

            nearby_nodes = target_nodes[target_nodes.geometry.distance(source_node_geometry) < distance_threshold]
            for _, target_node in nearby_nodes.iterrows():
                # Transform the target node geometry to CRS 4087
                target_node_geometry = target_node.geometry  # .to_crs("EPSG:4087")

                connected_edge = gpd.GeoDataFrame(
                    {
                        'source': [source_node['osmid']],
                        'target': [target_node['osmid']],
                        'geometry': [LineString([source_node.geometry, target_node.geometry])]
                    }
                )
                # connected_edges = connected_edges.append(connected_edge)
                connected_edges = pd.concat([connected_edges, connected_edge], ignore_index=True)
    # Project the connected edges and merged nodes back to CRS 4326
    # connected_edges = connected_edges.to_crs("EPSG:4326")

    # merged_edges = merged_edges.append(connected_edges)
    merged_edges = pd.concat([merged_edges, connected_edges], ignore_index=True)
    merged_edges = merged_edges.to_crs("EPSG:4326")
    merged_nodes = merged_nodes.to_crs("EPSG:4326")

    merged_nodes["osmid"] = range(len(merged_nodes["osmid"]))
    merged_edges = assignEndpoints(merged_edges, merged_nodes)
    merged_edges = merged_edges[["u", "v", "geometry"]]

    return merged_nodes, merged_edges


def addKm(edges, crs):
    # Project the layer. Watch out, the CRS should be adapted to the country
    edges['km'] = edges.to_crs({'init': 'epsg:' + str(crs)}).length / 1000
    return edges


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

def merge_networks(input_folder, bloc_name = "bloc", threshold=None):
    logging.info("MERGING NETWORKS")
    # input_folder = os.path.join(os.path.dirname(input_folder), "simp_B")
    gdf_edges = [file for file in os.listdir(os.path.join(input_folder)) if
                 (os.path.isfile(os.path.join(os.path.join(input_folder), file)) & ("edges" in file))]
    gdf_nodes = [file for file in os.listdir(os.path.join(input_folder)) if
                 (os.path.isfile(os.path.join(os.path.join(input_folder), file)) & ("nodes" in file))]

    logging.info("Merging: " + str(gdf_edges))

    geodataframes = []

    for i in range(len(gdf_edges)):
        # nodes = read_dataframe(os.path.join(input_folder, gdf_nodes[i]))
        # edges = read_dataframe(os.path.join(input_folder, gdf_edges[i]))
        # Masquer temporairement les avertissements de Fiona lors de la lecture du fichier
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="fiona")
            nodes = gpd.read_file(os.path.join(input_folder, gdf_nodes[i]))#, ignore_fields=["osmid_original"])
            edges = gpd.read_file(os.path.join(input_folder, gdf_edges[i]))#, ignore_fields=["osmid_original"])
            geodataframes.append((nodes, edges))

    merged_nodes, merged_edges = merge_geodataframes(geodataframes, distance_threshold=threshold)
    merged_edges = addKm(merged_edges, 4087)

    merged_nodes = merged_nodes.drop(columns=["osmid", "id"])
    merged_nodes = merged_nodes.reset_index().rename(columns={"index": "osmid"})
    merged_edges = assignEndpoints(merged_edges, merged_nodes)
    merged_edges = merged_edges.reset_index().rename(columns={"index": "id"})
    merged_edges["id"] = merged_edges.index
    merged_nodes = merged_nodes.rename(columns={"osmid": "id"})

    output_folder = os.path.join(os.path.dirname(input_folder), "final")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    write_dataframe(merged_nodes, os.path.join(output_folder, bloc_name + '_nodes.geojson'))
    write_dataframe(merged_edges, os.path.join(output_folder, bloc_name + '_edges.geojson'))
