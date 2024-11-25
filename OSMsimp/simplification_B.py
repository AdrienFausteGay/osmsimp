import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString
import shapely
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
import os
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from shapely.geometry import Point, MultiPoint, LineString
import progressbar
from sklearn.neighbors import BallTree
import shutil
from scipy.spatial.distance import pdist
import pyproj
from OSMsimp.makegraphconnected import make_graph_connected
from concurrent.futures import ThreadPoolExecutor
from pyogrio import read_dataframe
from shapely.ops import split
import itertools

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
    crs = edges.crs  # Stocker le CRS initial
    # identify start edge
    edge_ids_to_merge = identify_edges_connected_to_one_node(degree2_node_id, edges)
    if len(edge_ids_to_merge) != 2:
        logging.debug(f"Nodes {degree2_node_id} - there should be exactly two edges to merge, found {len(edge_ids_to_merge)}")
            # Suppression de l'unique arête trouvée
        edge_osmid_to_remove = edge_ids_to_merge[0]  # Obtenir l'ID 'osmid' de l'arête à supprimer
        edges = edges[edges['osmid'] != edge_osmid_to_remove]  # Supprimer la ligne correspondant à cet 'osmid'
        logging.debug(f"Removed the only edge for node {degree2_node_id} with osmid {edge_osmid_to_remove}")
        result = {"success": True,
                  "log": f"Removed the only edge for node {degree2_node_id} with osmid {edge_osmid_to_remove}"}
        return edges, result
    else:
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
            "log": f"Nodes {degree2_node_id} - merging failed, end_ids is not valid {end_ids}, delete one node connected to this node",
            "resulting_geometry": merged_edge_geom
        }
        edges.drop(edge_ids_to_merge[0], inplace=True)
        return edges, result
    # add the new merged edge
    new_row = merged_edge.to_frame().T
    new_row.index = [edges.index.max() + 1]
    edges = pd.concat([edges, new_row], axis=0)
    # remove the two edges
    edges = edges.drop(edge_ids_to_merge, axis=0)
    # print('removing '+str(edge_ids_to_merge))
    # print('new edge added with endpoints '+str(merged_edge[['u', 'v']].to_list()))
    # Réassigner le CRS au GeoDataFrame résultant après concaténation
    edges = edges.set_crs(crs)
    return edges, {"success": True}


def print_nb_edges_nodes(edges, nodes):
    logging.debug(f"There are {edges.shape[0]} edges and {nodes.shape[0]} nodes")

def project_to_utm(gdf):
    """
    Projette un GeoDataFrame en coordonnées géographiques (latitude/longitude) vers le système de coordonnées UTM approprié.

    :param gdf: GeoDataFrame en coordonnées géographiques (CRS WGS84)
    :return: GeoDataFrame projeté en UTM, code EPSG utilisé
    """
    # Vérifier que le CRS actuel est WGS84 (EPSG:4326)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Calculer le centroïde de l'ensemble des données
    centroid = gdf.unary_union.centroid
    lon, lat = centroid.x, centroid.y

    # Calculer la zone UTM
    utm_zone = int((lon + 180) / 6) % 60 + 1  # Modulo pour gérer les longitudes négatives

    # Déterminer l'hémisphère
    if lat >= 0:
        hemisphere = 'north'
        epsg_code = 32600 + utm_zone  # EPSG pour l'hémisphère Nord
    else:
        hemisphere = 'south'
        epsg_code = 32700 + utm_zone  # EPSG pour l'hémisphère Sud

    # Projeter le GeoDataFrame
    gdf_utm = gdf.to_crs(epsg=epsg_code)

    return gdf_utm, epsg_code


def merge_close_points(df_nodes, df_edges, R, min_samples=2):
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
    #Projection dans un crs adapté
    df_nodes, epsg_code = project_to_utm(df_nodes)
    df_edges = df_edges.set_crs(epsg=4326)
    df_edges = df_edges.to_crs(epsg=epsg_code)
    # Convert the node geometries to 2D numpy array for clustering
    coords = np.vstack(df_nodes.geometry.apply(lambda x: (x.x, x.y)).values)

    # # Scale the coordinates for better clustering performance
    # scaler = StandardScaler().fit(coords)
    # coords_scaled = scaler.transform(coords)

    # Apply DBSCAN clustering to the scaled coordinates
    dbscan = DBSCAN(eps=R*1000, min_samples=min_samples, metric='euclidean', algorithm='kd_tree', n_jobs=1).fit(coords)

    # Assign cluster labels to the original node dataframe
    df_nodes_new = df_nodes.copy()
    df_nodes_new['cluster'] = dbscan.labels_

    # Merge nodes within each cluster by computing the centroid of each cluster
    # and creating a new GeoDataFrame with the merged Point geometries
    new_geoms = []
    new_index = []

    # Ajouter une colonne pour conserver la géométrie originale des nœuds
    df_nodes['original_geometry'] = df_nodes['geometry']

    # Copier df_nodes pour travailler sur une nouvelle version
    df_nodes_new = df_nodes.copy()

    # Supposons que 'dbscan.labels_' contienne les étiquettes des clusters après clustering
    # Ajouter les labels de cluster aux nœuds
    df_nodes_new['cluster'] = dbscan.labels_

    # Initialiser les listes pour les nouveaux nœuds (centroïdes)
    new_geoms = []
    new_index = []

    # Parcourir les clusters pour fusionner les nœuds
    for i in tqdm(np.unique(dbscan.labels_), desc='Merging nodes'):
        if i == -1:
            # Nœuds qui ne font pas partie d'un cluster
            df_nodes_new.loc[df_nodes_new['cluster'] == i, 'new index'] = df_nodes_new.loc[df_nodes_new['cluster'] == i, 'osmid']
        else:
            # Nœuds à fusionner
            cluster_nodes = df_nodes_new.loc[df_nodes_new['cluster'] == i]
            new_node_id = int(i + len(df_nodes_new))
            df_nodes_new.loc[df_nodes_new['cluster'] == i, 'new index'] = new_node_id
            # Calculer le centroïde du cluster
            cluster_points = MultiPoint(cluster_nodes['geometry'].tolist())
            centroid = cluster_points.centroid
            new_geoms.append(centroid)
            new_index.append(new_node_id)

    # Créer un GeoDataFrame pour les nouveaux nœuds (centroïdes)
    d = {
        "id": new_index,
        "new index": new_index,
        "cluster": [-1 for _ in range(len(new_index))],
        "geometry": new_geoms,
        "original_geometry": new_geoms  # Pour les centroïdes, la géométrie originale est le centroïde
    }
    merged_nodes = gpd.GeoDataFrame(d, crs=df_nodes.crs)

    # Combiner les nœuds originaux et les nouveaux nœuds (centroïdes)
    new_nodes = pd.concat([df_nodes_new, merged_nodes], ignore_index=True)

    # Assurer que les types de données sont corrects
    new_nodes = new_nodes.astype({"new index": int})
    new_nodes['osmid'].fillna(-1, inplace=True)
    new_nodes = new_nodes.astype({"osmid": int})

    # Copier df_edges pour travailler sur une nouvelle version
    df_edges_new = df_edges.copy()

    # Créer un dictionnaire de mappage des anciens identifiants de nœuds vers les nouveaux
    node_map = dict(zip(new_nodes['osmid'], new_nodes['new index']))
    #Suppression des noeuds qui ont été fusionné
    new_nodes = new_nodes.loc[new_nodes["cluster"] == -1].copy()
    # Mettre à jour les colonnes 'u' et 'v' avec les nouveaux identifiants
    df_edges_new['u'] = df_edges_new['u'].map(node_map).fillna(df_edges_new['u'])
    df_edges_new['v'] = df_edges_new['v'].map(node_map).fillna(df_edges_new['v'])

    # Préparer une liste pour les nouvelles géométries des arêtes
    new_geometries = []

    # Créer une barre de progression
    bar = progressbar.ProgressBar(maxval=len(df_edges_new), 
                                widgets=[
                                    'Mise à jour des arêtes : ', progressbar.SimpleProgress(),
                                    ' (', progressbar.Percentage(), ') ',
                                    progressbar.Bar(), ' ', progressbar.ETA()
                                ])

    # Parcourir les arêtes pour ajuster les géométries
    for idx, row in bar(df_edges_new.iterrows()):
        u = row['u']
        v = row['v']
        edge_geom = row['geometry']

        # Récupérer les informations des nœuds 'u' et 'v'
        u_row = new_nodes.loc[new_nodes['new index'] == u].iloc[0]
        v_row = new_nodes.loc[new_nodes['new index'] == v].iloc[0]

        # Déterminer si les nœuds ont été fusionnés
        u_has_moved = u_row['osmid'] != u_row['new index']
        v_has_moved = v_row['osmid'] != v_row['new index']

        # Géométries nouvelles et originales des nœuds
        u_new_geom = u_row['geometry']
        v_new_geom = v_row['geometry']
        u_original_geom = u_row['original_geometry']
        v_original_geom = v_row['original_geometry']

        # Fonction pour ajuster la géométrie de l'arête
        def adjust_edge_geometry(edge_geometry, u_old_geom, u_new_geom, u_has_moved, 
                                v_old_geom, v_new_geom, v_has_moved):
            coords = list(edge_geometry.coords)
            # Ajuster le point de départ
            if u_has_moved:
                coords[0] = (u_new_geom.x, u_new_geom.y)
            # Ajuster le point d'arrivée
            if v_has_moved:
                coords[-1] = (v_new_geom.x, v_new_geom.y)
            return LineString(coords)

        # Ajuster la géométrie de l'arête
        adjusted_geom = adjust_edge_geometry(edge_geom, u_original_geom, u_new_geom, u_has_moved,
                                            v_original_geom, v_new_geom, v_has_moved)
        new_geometries.append(adjusted_geom)
        bar.update(idx + 1)

    # Mettre à jour les géométries des arêtes
    df_edges_new['geometry'] = new_geometries

    # for i in  tqdm(np.unique(dbscan.labels_), desc='Merging nodes'):
    #     if i == -1:
    #         df_nodes_new.loc[df_nodes_new['cluster'] == i, 'new index'] = df_nodes_new.loc[
    #             df_nodes_new['cluster'] == i, 'osmid']
    #     else:
    #         cluster_points = df_nodes_new.loc[df_nodes_new['cluster'] == i, 'geometry']
    #         df_nodes_new.loc[df_nodes_new['cluster'] == i, 'new index'] = int(i + len(df_nodes_new))
    #         cluster_points = MultiPoint([Point(xy) for xy in cluster_points.geometry.values])
    #         centroid = cluster_points.centroid
    #         new_geoms.append(centroid)
    #         new_index.append(len(df_nodes_new) + i)
    # d = {"id": new_index, "new index": new_index, "cluster": [-1 for i in range(len(new_index))], "geometry": new_geoms}
    # merged_nodes = gpd.GeoDataFrame(d, crs=df_nodes.crs)  # geometry=new_geoms
    # new_nodes = pd.concat([df_nodes_new, merged_nodes])  # df_nodes_new.append(merged_nodes)
    # new_nodes = new_nodes.astype({"new index": int})
    # new_nodes['osmid'].fillna(-1, inplace=True)
    # new_nodes = new_nodes.astype({"osmid": int})

    # df_edges_new = df_edges.copy()

    # node_map = dict(zip(new_nodes['osmid'], new_nodes['new index']))
    # df_edges_new['u'] = df_edges_new['u'].map(node_map).fillna(df_edges_new['u'])
    # df_edges_new['v'] = df_edges_new['v'].map(node_map).fillna(df_edges_new['v'])

    # new_nodes_t = new_nodes.loc[new_nodes["cluster"] == -1].copy()

    # # Créer la barre de progression avec progressbar2
    # bar = progressbar.ProgressBar(maxval=len(df_edges_new), 
    #                           widgets=[
    #                               'Fusion des clusters : ', progressbar.SimpleProgress(),
    #                               ' (', progressbar.Percentage(), ') ',
    #                               progressbar.Bar(), ' ', progressbar.ETA()
    #                           ])
    
    # # Préparer une liste pour les nouvelles géométries
    # new_geometries = []

    # # Utiliser la barre de progression dans la boucle
    # for idx, (u, v) in enumerate(bar(zip(df_edges_new['u'], df_edges_new['v']))):
    #     geom_u = new_nodes_t.loc[new_nodes_t["new index"] == u, "geometry"].values[0]
    #     geom_v = new_nodes_t.loc[new_nodes_t["new index"] == v, "geometry"].values[0]
    #     line = LineString([geom_u, geom_v])
    #     new_geometries.append(line)
    #     bar.update(idx + 1)
    # # Préparer une liste pour les nouvelles géométries
    # df_edges_new['geometry'] = new_geometries
    # new_nodes_t.drop(columns=['osmid'], inplace=True)
    # new_nodes_t = new_nodes_t.rename(columns={'new index': 'osmid'})
    new_nodes.drop(columns=['osmid', 'original_geometry', 'osmid_original'], inplace=True)
    new_nodes = new_nodes.rename(columns={'new index': 'osmid'})
    return new_nodes, df_edges_new

def split_line_at_point(line, point):
    # # Vérifier que le point est bien de type Point et sur la ligne
    # if not isinstance(point, Point) or not (line.contains(point) or line.touches(point)):
    #     # Si le point n'est pas sur la ligne, on retourne la ligne sans la diviser
    #     return [line]

    # Trouver la distance le long de la ligne jusqu'au point d'intersection
    distance = line.project(point)

    # Vérifier si le point est au début ou à la fin du LineString, alors pas de division nécessaire
    if distance <= 0.0 or distance >= line.length:
        return [line]

    # Interpoler le point d'intersection pour garantir qu'il est sur la ligne
    interpolated_point = line.interpolate(distance)
    coords = list(line.coords)

    # Créer deux nouveaux segments en divisant au point interpolé
    segment1 = []
    segment2 = []
    point_added = False

    for coord in coords:
        if not point_added and LineString([coord, coords[coords.index(coord)+1]]).distance(interpolated_point) < 1e-6:
            # Ajouter jusqu'à l'interpolated_point pour le premier segment
            segment1.append(coord)
            segment1.append((interpolated_point.x, interpolated_point.y))
            segment2.append((interpolated_point.x, interpolated_point.y))
            point_added = True
        elif not point_added:
            segment1.append(coord)
        else:
            segment2.append(coord)

    # Créer les nouveaux LineStrings à partir des segments
    line1 = LineString(segment1)
    line2 = LineString(segment2)

    return [line1, line2]

def planarize_graph(nodes, edges):
    """
    Planarise un graphe en ajoutant des nœuds aux intersections d'arêtes
    qui ne sont pas déjà des nœuds. Divise les arêtes aux points d'intersection
    et met à jour les colonnes 'u' et 'v'.
    
    Paramètres :
    - nodes : GeoDataFrame des nœuds avec une colonne 'geometry' de Points et un identifiant 'id'.
    - edges : GeoDataFrame des arêtes avec des colonnes 'u', 'v', et 'geometry' (LineString).
    
    Retourne :
    - new_nodes : GeoDataFrame des nœuds mis à jour avec les nouveaux nœuds ajoutés.
    - new_edges : GeoDataFrame des arêtes mises à jour avec les arêtes divisées et les colonnes 'u' et 'v' mises à jour.
    """
    
    # Copier les nœuds et arêtes pour ne pas modifier les originaux
    nodes = nodes.copy()
    edges = edges.copy()

    # Liste pour stocker les nouvelles arêtes divisées
    new_edges = []
    
    # Variable pour générer des identifiants uniques pour les nouveaux nœuds
    max_node_id = nodes['osmid'].max() if 'id' in nodes.columns else nodes.index.max()
    
    # Détection des intersections entre toutes les paires d'arêtes
    for i, edge1 in edges.iterrows():
        for j, edge2 in edges.iterrows():
            if i >= j:
                continue  # Éviter les doublons et les auto-intersections

            # Vérifier s'il y a une intersection entre les deux arêtes
            intersection = edge1.geometry.intersection(edge2.geometry)
            
            # Vérifier si l'intersection est non vide et est un Point ou MultiPoint
            if isinstance(intersection, (Point, MultiPoint)) and not intersection.is_empty:
                # Si c'est un seul point, le convertir en liste pour un traitement uniforme
                intersection_points = [intersection] if isinstance(intersection, Point) else list(intersection.geoms)

                # Parcourir chaque point d'intersection
                for point in intersection_points:
                    # Vérifier si le point est aux extrémités de l'une des arêtes
                    if point.equals(Point(edge1.geometry.coords[0])) or point.equals(Point(edge1.geometry.coords[-1])) or \
                    point.equals(Point(edge2.geometry.coords[0])) or point.equals(Point(edge2.geometry.coords[-1])):
                        continue  # Ignorer les points aux extrémités
                    # Ajouter l'intersection comme nouveau nœud
                    max_node_id += 1
                    new_node = {'id': max_node_id, 'geometry': point}
                    nodes = pd.concat([nodes, gpd.GeoDataFrame([new_node], crs=nodes.crs)], ignore_index=True)

                    # Diviser les deux arêtes à l'intersection
                    split_edge1_segments = split_line_at_point(edge1.geometry, point)
                    split_edge2_segments = split_line_at_point(edge2.geometry, point)

                    # Créer de nouvelles arêtes pour edge1 divisée
                    u = edge1['u']
                    for segment in split_edge1_segments:
                        if not segment.is_empty:
                            # Déterminer v
                            if segment.coords[-1] == (point.x, point.y):
                                v = max_node_id
                            else:
                                v = edge1['v']
                            new_edges.append({'u': u, 'v': v, 'geometry': segment})
                            u = v  # Mettre à jour u pour le prochain segment

                    # Faire de même pour edge2
                    u = edge2['u']
                    for segment in split_edge2_segments:
                        if not segment.is_empty:
                            if segment.coords[-1] == (point.x, point.y):
                                v = max_node_id
                            else:
                                v = edge2['v']
                            new_edges.append({'u': u, 'v': v, 'geometry': segment})
                            u = v

            else:
                # Si pas d'intersection, conserver les arêtes originales
                new_edges.append(edge1)
                new_edges.append(edge2)

    # Créer le GeoDataFrame final des arêtes divisées avec u et v mis à jour
    new_edges = [edge.to_dict() if isinstance(edge, pd.Series) else edge for edge in new_edges]
    new_edges_df = gpd.GeoDataFrame(new_edges, geometry='geometry', crs=edges.crs)
    
    # Enlever les doublons d'arêtes
    new_edges_df = new_edges_df.drop_duplicates(subset=['geometry'])
    nodes['osmid'] = nodes['osmid'].fillna(nodes['id'])
    nodes['id'] = nodes['id'].fillna(nodes['osmid'])

    return nodes.reset_index(drop=True), new_edges_df.reset_index(drop=True)

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
    N_same_start_end = len(edges[boolean_same_start_end])
    # print(f'Removing {boolean_same_start_end.sum()} edges with same start and end nodes')
    edges = edges[~boolean_same_start_end]
    # print_nb_edges_nodes(edges, nodes)

    # Remove nodes that are nowhere (can happen after the previous step)
    boolean_useless_nodes = ~nodes['osmid'].isin(edges['u']) & ~nodes['osmid'].isin(edges['v'])
    # print(f'Removing {boolean_useless_nodes.sum()} nodes attached to no edges')
    nodes = nodes[~boolean_useless_nodes]
    # print_nb_edges_nodes(edges, nodes)

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
    return edges, nodes, len(degree2_nodes), N_same_start_end


def simplification_B(files_folder, R=5, correction_connexe=True, output_folder=None):
    logging.info("SIMPLIFICATION B PHASE")
    # files_folder = os.path.join(os.path.dirname(files_folder), "simp_A")
    files = set([file for file in os.listdir(os.path.join(files_folder)) if
                 os.path.isfile(os.path.join(os.path.join(files_folder), file))])

    for i in files:
        edges = read_dataframe(
            os.path.join(files_folder, i), layer=1)
        nodes = read_dataframe(
            os.path.join(files_folder, i), layer=0)

        logging.info("Working on: " + str(i[:-5]))
        print_nb_edges_nodes(edges, nodes)
        if len(nodes) < 10:
            #C'est pas très propre comme critère... En fait le graphe est réduit à 0 si il n'est fait que de noeuds de degre 2 ou 3...
            # Cast types (seems to create pb in exporting otherwise)
            edges['u'] = edges['u'].astype(int)
            edges['v'] = edges['v'].astype(int)
            nodes['osmid'] = nodes['osmid'].astype(int)

            # Remove edges which have the same start and end nodes (seems to happen in edges that are disconnected from the main network)
            boolean_same_start_end = edges['u'] == edges['v']
            N_same_start_end = len(edges[boolean_same_start_end])
            # print(f'Removing {boolean_same_start_end.sum()} edges with same start and end nodes')
            edges = edges[~boolean_same_start_end]
            print_nb_edges_nodes(edges, nodes)
        else:
            N_same_start_end = 1
            num_degree_2 = 1
            while num_degree_2 > 0 or N_same_start_end > 0:
                edges, nodes, num_degree_2, N_same_start_end = remove_useless_nodes(edges, nodes)

            nodes["osmid"] = range(len(nodes["osmid"]))
            edges = assignEndpoints(edges, nodes)
            if correction_connexe:
                nodes, edges = make_graph_connected(nodes, edges, 25e3)

            nodes, edges = merge_close_points(nodes, edges, R=R, min_samples=2)

        if len(nodes) < 10: #gestion du cas où la simplification est si forte qu'il ne reste qu'une quelques noeuds
            #C'est pas très propre comme critère... En fait le graphe est réduit à 0 si il n'est fait que de noeuds de degre 2 ou 3...
            # Cast types (seems to create pb in exporting otherwise)
            edges['u'] = edges['u'].astype(int)
            edges['v'] = edges['v'].astype(int)
            nodes['osmid'] = nodes['osmid'].astype(int)

            # Remove edges which have the same start and end nodes (seems to happen in edges that are disconnected from the main network)
            boolean_same_start_end = edges['u'] == edges['v']
            N_same_start_end = len(edges[boolean_same_start_end])
            # print(f'Removing {boolean_same_start_end.sum()} edges with same start and end nodes')
            edges = edges[~boolean_same_start_end]
            print_nb_edges_nodes(edges, nodes)

        else:
            print_nb_edges_nodes(edges, nodes)
            num_degree_2 = 1
            N_same_start_end = 1
            while num_degree_2 > 0 or N_same_start_end > 0:
                if len(nodes)<10:
                    break
                edges, nodes, _, _ = remove_useless_nodes(edges, nodes)
                if len(nodes)<10:
                    break
                logging.debug("removing duplicate rows")
                edges = remove_duplicate_rows(edges)
                nodes, edges = planarize_graph(nodes, edges)
                edges, nodes, num_degree_2, N_same_start_end = remove_useless_nodes(edges, nodes)

        # Export
        edges['u'] = edges['u'].astype(int)  # recast, create a bug otherwise
        edges['v'] = edges['v'].astype(int)
        nodes['osmid'] = nodes['osmid'].astype(int)
        if output_folder == None:
            output_folder = os.path.join(os.path.dirname(files_folder), "simp_B")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        nodes.reset_index(drop=True).to_file(
            os.path.join(output_folder, i[:-5] + '_nodes.geojson'), driver="GeoJSON")
        edges[['u', 'v', 'highway', 'geometry']].reset_index(drop=True).to_file(
            os.path.join(output_folder, i[:-5] + '_edges.geojson'), driver="GeoJSON")
        if not os.path.exists(os.path.join(files_folder, "Done")):
            os.makedirs(os.path.join(files_folder, "Done"))
        # shutil.move(os.path.join(files_folder, i), os.path.join(files_folder, "Done", i))


#Version parralélisée

def process_file(file, files_folder, R, correction_connexe, output_folder):
    edges = gpd.read_file(os.path.join(files_folder, file), layer=1)
    nodes = gpd.read_file(os.path.join(files_folder, file), layer=0)

    logging.info("Working on: " + str(file[:-5]))
    print_nb_edges_nodes(edges, nodes)

    N_same_start_end = 1
    num_degree_2 = 1
    while num_degree_2 > 0 or N_same_start_end > 0:
        edges, nodes, num_degree_2, N_same_start_end = remove_useless_nodes(edges, nodes)

    nodes["osmid"] = range(len(nodes["osmid"]))
    edges = assignEndpoints(edges, nodes)
    
    if correction_connexe:
        nodes, edges = make_graph_connected(nodes, edges, 25e3)

    nodes, edges = merge_close_points(nodes, edges, R=R, min_samples=2)

    if len(nodes) < 10:
        edges['u'] = edges['u'].astype(int)
        edges['v'] = edges['v'].astype(int)
        nodes['osmid'] = nodes['osmid'].astype(int)
        boolean_same_start_end = edges['u'] == edges['v']
        N_same_start_end = len(edges[boolean_same_start_end])
        edges = edges[~boolean_same_start_end]
        print_nb_edges_nodes(edges, nodes)

    else:
        print_nb_edges_nodes(edges, nodes)
        num_degree_2 = 1
        N_same_start_end = 1
        while num_degree_2 > 0 or N_same_start_end > 0:
            edges, nodes, _, _ = remove_useless_nodes(edges, nodes)
            logging.debug("removing duplicate rows")
            edges = remove_duplicate_rows(edges)
            edges, nodes, num_degree_2, N_same_start_end = remove_useless_nodes(edges, nodes)

    edges['u'] = edges['u'].astype(int)
    edges['v'] = edges['v'].astype(int)
    nodes['osmid'] = nodes['osmid'].astype(int)
    if output_folder==None:
        output_folder = os.path.join(os.path.dirname(files_folder), "simp_B")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nodes.reset_index(drop=True).to_file(os.path.join(output_folder, file[:-5] + '_nodes.geojson'), driver="GeoJSON")
    edges[['u', 'v', 'highway', 'geometry']].reset_index(drop=True).to_file(
        os.path.join(output_folder, file[:-5] + '_edges.geojson'), driver="GeoJSON")
    
    if not os.path.exists(os.path.join(files_folder, "Done")):
        os.makedirs(os.path.join(files_folder, "Done"))
    shutil.move(os.path.join(files_folder, file), os.path.join(files_folder, "Done", file))

def simplification_B_par(files_folder, R=30, correction_connexe=True, output_folder=None):
    logging.info("SIMPLIFICATION B PHASE")
    files = {file for file in os.listdir(os.path.join(files_folder)) if os.path.isfile(os.path.join(files_folder, file))}

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [
            executor.submit(process_file, file, files_folder, R, correction_connexe, output_folder)
            for file in files
        ]
        
        for future in futures:
            future.result()  # Attend que chaque tâche se termine pour gérer les exceptions éventuelles