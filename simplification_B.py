import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString
import shapely
from tqdm import tqdm
import logging

import os
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from shapely.geometry import Point, MultiPoint, LineString
import progressbar
from sklearn.neighbors import BallTree
import shutil


def get_degree1_nodes(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> list:
    node_occurrence_as_start_end = pd.concat([edges['u'], edges['v']], axis=0, ignore_index=True).value_counts()
    return node_occurrence_as_start_end[node_occurrence_as_start_end == 1].index.to_list()


def get_degree2_nodes(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> list:
    node_occurrence_as_start_end = pd.concat([edges['u'], edges['v']], axis=0, ignore_index=True).value_counts()
    return node_occurrence_as_start_end[node_occurrence_as_start_end == 2].index.to_list()


def identify_edges_connected_to_one_node(node_id: int, edges: gpd.GeoDataFrame) -> list:
    boolean_start = edges['u'] == node_id
    boolean_end = edges['v'] == node_id
    return edges[boolean_start | boolean_end].index.to_list()


def remove_degree2_node_and_merge(degree2_node_id: int, edges: gpd.GeoDataFrame):
    # identify start edge
    edge_ids_to_merge = identify_edges_connected_to_one_node(degree2_node_id, edges)
    if len(edge_ids_to_merge) != 2:
        raise ValueError(
            f"Nodes {degree2_node_id} - there should be exactly two edges to merge, found {len(edge_ids_to_merge)}")
    # merge the two edges
    multi_line = shapely.geometry.MultiLineString(
        [edges.loc[edge_ids_to_merge[0], "geometry"], edges.loc[edge_ids_to_merge[1], "geometry"]]
    )
    merged_edge_geom = shapely.ops.linemerge(multi_line)
    if merged_edge_geom.geom_type != 'LineString':
        result = {
            "success": False,
            "log": f"Nodes {degree2_node_id} - merging failed, expected to produce a LineString, \
                   produced a {merged_edge_geom.geom_type}, do nothing for this node",
            "resulting_geometry": merged_edge_geom
        }
        return edges, result
    # create new edge row based on the first edge to merge (arbitrary)
    merged_edge = edges.loc[edge_ids_to_merge[0]].copy(deep=True)
    merged_edge['geometry'] = merged_edge_geom
    # add the adequate u anv d node_ids
    end_ids = edges.loc[edge_ids_to_merge[0], ['u', 'v']].to_list() + edges.loc[
        edge_ids_to_merge[1], ['u', 'v']].to_list()
    # print(set(end_ids), {degree2_node_id}, list(set(end_ids) - {degree2_node_id}))
    end_ids = list(set(end_ids) - {degree2_node_id})
    try:
        merged_edge['u'] = end_ids[0]
        merged_edge['v'] = end_ids[1]
    except:
        result = {
            "success": False,
            "log": f"Nodes {degree2_node_id} - merging failed, end_ids is not valid {end_ids}, do nothing for this node",
            "resulting_geometry": merged_edge_geom
        }
        return edges, result
    # add the new merged edge
    new_row = merged_edge.to_frame().T
    new_row.index = [edges.index.max() + 1]
    edges = pd.concat([edges, new_row], axis=0)
    # remove the two edges
    edges = edges.drop(edge_ids_to_merge, axis=0)
    # print('removing '+str(edge_ids_to_merge))
    # print('new edge added with endpoints '+str(merged_edge[['u', 'v']].to_list()))
    return edges, {"success": True}


def print_nb_edges_nodes(edges, nodes):
    logging.info(f"There are {edges.shape[0]} edges and {nodes.shape[0]} nodes")


def merge_close_points(df_nodes, df_edges, eps=0.001, min_samples=2):
    """
    Given GeoDataFrames with Point geometries for nodes and LineString geometries for edges,
    merges nearby nodes within a given distance threshold using DBSCAN clustering algorithm,
    and updates the edges' endpoints accordingly.

    :param df_nodes: GeoDataFrame with Point geometries for nodes
    :param df_edges: GeoDataFrame with LineString geometries for edges
    :param eps: DBSCAN epsilon value (default: 0.0001)
    :param min_samples: DBSCAN minimum number of samples (default: 2)
    :return: Tuple of GeoDataFrames with merged Point geometries for nodes and updated endpoints for edges
    """
    # Convert the node geometries to 2D numpy array for clustering
    coords = np.vstack(df_nodes.geometry.apply(lambda x: (x.x, x.y)).values)

    # Scale the coordinates for better clustering performance
    scaler = StandardScaler().fit(coords)
    coords_scaled = scaler.transform(coords)

    # Apply DBSCAN clustering to the scaled coordinates
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='auto').fit(coords_scaled)

    # Assign cluster labels to the original node dataframe
    df_nodes_new = df_nodes.copy()
    df_nodes_new['cluster'] = dbscan.labels_

    # Merge nodes within each cluster by computing the centroid of each cluster
    # and creating a new GeoDataFrame with the merged Point geometries
    new_geoms = []
    new_index = []
    for i in tqdm(np.unique(dbscan.labels_), desc='Merging nodes'):
        if i == -1:
            df_nodes_new.loc[df_nodes_new['cluster'] == i, 'new index'] = df_nodes_new.loc[
                df_nodes_new['cluster'] == i, 'osmid']
        else:
            cluster_points = df_nodes_new.loc[df_nodes_new['cluster'] == i, 'geometry']
            df_nodes_new.loc[df_nodes_new['cluster'] == i, 'new index'] = int(i + len(df_nodes_new))
            cluster_points = MultiPoint([Point(xy) for xy in cluster_points.geometry.values])
            centroid = cluster_points.centroid
            new_geoms.append(centroid)
            new_index.append(len(df_nodes_new) + i)
    d = {"id": new_index, "new index": new_index, "cluster": [-1 for i in range(len(new_index))], "geometry": new_geoms}
    merged_nodes = gpd.GeoDataFrame(d, crs=df_nodes.crs)  # geometry=new_geoms
    new_nodes = pd.concat([df_nodes_new, merged_nodes])  # df_nodes_new.append(merged_nodes)
    new_nodes = new_nodes.astype({"new index": int})
    new_nodes['osmid'].fillna(-1, inplace=True)
    new_nodes = new_nodes.astype({"osmid": int})

    df_edges_new = df_edges.copy()

    node_map = dict(zip(new_nodes['osmid'], new_nodes['new index']))
    df_edges_new['u'] = df_edges_new['u'].map(node_map).fillna(df_edges_new['u'])
    df_edges_new['v'] = df_edges_new['v'].map(node_map).fillna(df_edges_new['v'])

    new_nodes_t = new_nodes.loc[new_nodes["cluster"] == -1]

    df_edges_new['geometry'] = [
        LineString([
            new_nodes_t.loc[new_nodes_t["new index"] == u, "geometry"].values[0],
            new_nodes_t.loc[new_nodes_t["new index"] == v, "geometry"].values[0]
        ])
        for u, v in progressbar.progressbar(zip(df_edges_new['u'], df_edges_new['v']), max_value=len(df_edges_new))
    ]
    new_nodes_t.drop(columns=['osmid'], inplace=True)
    new_nodes_t = new_nodes_t.rename(columns={'new index': 'osmid'})
    return new_nodes_t, df_edges_new


def remove_duplicate_rows(gdf):
    # Create a copy of the GeoDataFrame
    unique_gdf = gdf.copy()

    # Sort 'u' and 'v' columns to make them interchangeable
    unique_gdf[['u', 'v']] = unique_gdf[['u', 'v']].apply(lambda row: sorted(row), axis=1, result_type='expand')

    # Drop duplicate rows based on 'u' and 'v' columns
    unique_gdf = unique_gdf.drop_duplicates(subset=['u', 'v'])

    return unique_gdf


def getEndPointsFromLine(linestring_obj):
    if isinstance(linestring_obj, MultiLineString):
        linestring_obj = linestring_obj.geoms[0]
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
    df_links['u'] = idClosestPointsArray[:, 0]
    df_links['v'] = idClosestPointsArray[:, 1]
    return df_links.astype({'u': int, 'v': int})


def addKm(edges, crs):
    # Project the layer. Watch out, the CRS should be adapted to the country
    edges['km'] = edges.to_crs({'init': 'epsg:' + str(crs)}).length / 1000
    return edges


def remove_useless_nodes(edges, nodes):
    # Cast types (seems to create pb in exporting otherwise)
    edges['u'] = edges['u'].astype(int)
    edges['v'] = edges['v'].astype(int)
    nodes['osmid'] = nodes['osmid'].astype(int)

    # Check all start and end nodes specific in u and v are in the nodes tables
    # print(edges['u'].isin(nodes['osmid']).all())
    # print(edges['v'].isin(nodes['osmid']).all())

    # Remove edges which have the same start and end nodes (seems to happen in edges that are disconnected from the main network)
    boolean_same_start_end = edges['u'] == edges['v']
    # print(f'Removing {boolean_same_start_end.sum()} edges with same start and end nodes')
    edges = edges[~boolean_same_start_end]
    print_nb_edges_nodes(edges, nodes)

    # Remove nodes that are nowhere (can happen after the previous step)
    boolean_useless_nodes = ~nodes['osmid'].isin(edges['u']) & ~nodes['osmid'].isin(edges['v'])
    # print(f'Removing {boolean_useless_nodes.sum()} nodes attached to no edges')
    nodes = nodes[~boolean_useless_nodes]
    print_nb_edges_nodes(edges, nodes)

    # Calculate nb of degree 2 nodes
    degree2_nodes = get_degree2_nodes(nodes, edges)
    logging.info(f"Removing {len(degree2_nodes)} degree-2 nodes")

    # Rearrange the edges without them
    node_ids_to_remove = []
    issues = []
    for node_to_treat in tqdm(degree2_nodes):
        # print(f'  - Dealing with node {node_to_treat}')
        edges, result = remove_degree2_node_and_merge(node_to_treat, edges)
        if result['success']:
            node_ids_to_remove += [node_to_treat]
        else:
            issues += [result['log']]
        # Removing the successully treated nodes from nodes
    nodes = nodes[~nodes['osmid'].isin(node_ids_to_remove)]

    # print(f"There are {len(get_degree2_nodes(nodes, edges))} degree-2 nodes left")
    # print_nb_edges_nodes(edges, nodes)

    # Remove edges that are connected to two terminal nodes
    terminal_nodes = get_degree1_nodes(nodes, edges)
    boolean_isolated_edges = edges['u'].isin(terminal_nodes) & edges['v'].isin(terminal_nodes)
    # print(f'Removing {boolean_isolated_edges.sum()} edges attached to only terminal nodes and those nodes')
    node_ids_to_remove = list(
        set(edges.loc[boolean_isolated_edges, 'u'].to_list() + edges.loc[boolean_isolated_edges, 'v'].to_list()))
    edges = edges[~boolean_isolated_edges]
    nodes = nodes[~nodes['osmid'].isin(node_ids_to_remove)]
    print_nb_edges_nodes(edges, nodes)
    return edges, nodes


def simplification_B(files_folder, epsi=None):
    logging.info("SIMPLIFICATION B PHASE")
    files_folder = os.path.join(os.path.dirname(files_folder), "simp_A")
    files = set([file for file in os.listdir(os.path.join(files_folder)) if
                 os.path.isfile(os.path.join(os.path.join(files_folder), file))])

    for i in files:
        edges = gpd.read_file(
            os.path.join(files_folder, i), layer=1)
        nodes = gpd.read_file(
            os.path.join(files_folder, i), layer=0)

        logging.info("Working on: " + str(i[:-5]))
        print_nb_edges_nodes(edges, nodes)

        edges, nodes = remove_useless_nodes(edges, nodes)

        nodes["osmid"] = range(len(nodes["osmid"]))
        edges = assignEndpoints(edges, nodes)

        if epsi == None:
            if len(nodes) < 400:
                epsi = 0.005
            else:
                epsi = 0.01

        logging.debug("epsi " + str(i) + " : " + str(epsi))

        nodes_simp, edges_simp = merge_close_points(nodes, edges, eps=epsi, min_samples=2)

        nodes = nodes_simp
        edges = edges_simp

        print_nb_edges_nodes(edges, nodes)

        edges, nodes = remove_useless_nodes(edges, nodes)

        logging.debug("removing duplicate rows")
        edges = remove_duplicate_rows(edges)

        edges, nodes = remove_useless_nodes(edges, nodes)

        # Export
        edges['u'] = edges['u'].astype(int)  # recast, create a bug otherwise
        edges['v'] = edges['v'].astype(int)
        nodes['osmid'] = nodes['osmid'].astype(int)
        output_folder = os.path.join(os.path.dirname(files_folder), "simp_B")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        nodes.reset_index(drop=True).to_file(
            os.path.join(output_folder, i[:-5] + '_nodes.geojson'), driver="GeoJSON")
        edges[['u', 'v', 'highway', 'geometry']].reset_index(drop=True).to_file(
            os.path.join(output_folder, i[:-5] + '_edges.geojson'), driver="GeoJSON")
        if not os.path.exists(os.path.join(files_folder, "Done")):
            os.makedirs(os.path.join(files_folder, "Done"))
        shutil.move(os.path.join(files_folder, i), os.path.join(files_folder, "Done", i))
