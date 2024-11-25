import os
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from shapely.geometry import Point, LineString
from pyogrio import read_dataframe, write_dataframe
from sklearn.neighbors import BallTree
import logging
import warnings

def calculate_adaptive_threshold(nodes_1, nodes_2, lamb=1):
    # Convertir les géométries en tableau de coordonnées
    coords_1 = np.array([(point.x, point.y) for point in nodes_1.geometry])
    coords_2 = np.array([(point.x, point.y) for point in nodes_2.geometry])

    # Construire un BallTree pour l'ensemble de points de nodes_2
    tree = BallTree(coords_2, metric='euclidean')

    # Trouver la distance minimale pour chaque point de nodes_1 vers l'ensemble de nodes_2
    distances, _ = tree.query(coords_1, k=1)  # k=1 pour la plus proche distance
    
    # Extraire les distances minimales pour chaque point et calculer le 95e percentile
    threshold = np.percentile(distances, 1)
    
    return threshold*lamb

def merge_two_networks(network_a, network_b, distance_threshold):
    """
    Fusionne network_b dans network_a, en connectant les nœuds proches entre eux.
    """
    # S'assurer que les CRS sont cohérents
    network_a['nodes'] = network_a['nodes'].to_crs(crs='4087')
    network_a['edges'] = network_a['edges'].to_crs(crs='4087')
    network_b['nodes'] = network_b['nodes'].to_crs(crs='4087')
    network_b['edges'] = network_b['edges'].to_crs(crs='4087')

    # Ajuster les identifiants des nœuds de network_b
    max_node_id = network_a['nodes']['osmid'].max()
    network_b['nodes'] = network_b['nodes'].copy()
    network_b['edges'] = network_b['edges'].copy()
    network_b['nodes']['osmid'] = network_b['nodes']['osmid'] + max_node_id + 1
    # Mettre à jour 'u' et 'v' dans les arêtes
    if len(network_b['edges']) > 0:
        network_b['edges']['u'] = network_b['edges']['u'] + max_node_id + 1
        network_b['edges']['v'] = network_b['edges']['v'] + max_node_id + 1

    # Fusionner les nœuds et les arêtes
    merged_nodes = pd.concat([network_a['nodes'], network_b['nodes']], ignore_index=True)
    merged_edges = pd.concat([network_a['edges'], network_b['edges']], ignore_index=True)

    # Trouver les nœuds proches entre network_a et network_b
    nodes_a = network_a['nodes']
    nodes_b = network_b['nodes']
    coords_b = np.array([(geom.x, geom.y) for geom in nodes_b.geometry])
    tree_b = BallTree(coords_b, leaf_size=15, metric='euclidean')

    coords_a = np.array([(geom.x, geom.y) for geom in nodes_a.geometry])
    indices = tree_b.query_radius(coords_a, r=distance_threshold, return_distance=False)

    # Créer des arêtes entre les nœuds proches
    new_edges = []
    for idx_a, idxs_b in enumerate(indices):
        node_a_id = nodes_a.iloc[idx_a]['osmid']
        for idx_b in idxs_b:
            node_b_id = nodes_b.iloc[idx_b]['osmid']
            geom_a = nodes_a.iloc[idx_a].geometry
            geom_b = nodes_b.iloc[idx_b].geometry
            new_edge = {
                'u': node_a_id,
                'v': node_b_id,
                'geometry': LineString([geom_a, geom_b])
            }
            new_edges.append(new_edge)

    # Créer un GeoDataFrame pour les nouvelles arêtes
    new_edges_gdf = gpd.GeoDataFrame(new_edges, geometry='geometry', crs='4087')

    # Ajouter les nouvelles arêtes aux arêtes fusionnées
    merged_edges = pd.concat([merged_edges, new_edges_gdf], ignore_index=True)

    # Retourner le réseau fusionné
    merged_network = {
        'nodes': merged_nodes,
        'edges': merged_edges
    }

    return merged_network

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


def merge_networks(input_folder, bloc_name='bloc', lamb=1, start=None, distance_threshold=None):
    logging.info("MERGING NETWORKS")
    gdf_edges_files = sorted([file for file in os.listdir(input_folder) if
                     (os.path.isfile(os.path.join(input_folder, file)) and ("edges" in file))])
    gdf_nodes_files = sorted([file for file in os.listdir(input_folder) if
                     (os.path.isfile(os.path.join(input_folder, file)) and ("nodes" in file))])

    logging.info("Merging: " + str(gdf_edges_files))

    country_networks = []

    # Charger les pays et calculer les centroïdes
    for nodes_file, edges_file in zip(gdf_nodes_files, gdf_edges_files):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="fiona")
            nodes = gpd.read_file(os.path.join(input_folder, nodes_file))
            edges = gpd.read_file(os.path.join(input_folder, edges_file))
            nodes = nodes.to_crs(crs='4087')  # S'assurer d'un CRS cohérent
            centroid = nodes.geometry.unary_union.centroid
            country_networks.append({
                'name': nodes_file[:-14],  # Supposant que le nom du fichier se termine par '_nodes.geojson'
                'nodes': nodes,
                'edges': edges,
                'centroid': centroid,
                'merged': False
            })

    if start == None:
        # Choisir un pays de départ (par exemple, le premier)
        merged_network = {
            'nodes': country_networks[0]['nodes'],
            'edges': country_networks[0]['edges'],
        }
        country_networks[0]['merged'] = True
        i = 0
    else :
        i = next((index for index, country in enumerate(country_networks) if country['name'] == start), None)
        merged_network = {
            'nodes': country_networks[i]['nodes'],
            'edges': country_networks[i]['edges'],
        }
        country_networks[i]['merged'] = True
    logging.info(f"Starting merge with country: {country_networks[i]['name']}")

    # Fusion itérative des pays
    while True:
        unmerged_countries = [cn for cn in country_networks if not cn['merged']]
        if not unmerged_countries:
            break

        # Calculer le centroïde du réseau fusionné
        merged_centroid = merged_network['nodes'].geometry.unary_union.centroid

        # Trouver le pays non fusionné le plus proche
        min_distance = None
        closest_country = None
        for cn in unmerged_countries:
            distance = merged_centroid.distance(cn['centroid'])
            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest_country = cn

        # Vérifier si la distance est inférieure au seuil
        if distance_threshold == None:
            threshold = calculate_adaptive_threshold(merged_network['nodes'], cn['nodes'], lamb)
        if threshold is not None and min_distance > threshold:
            logging.info(f"Country {closest_country['name']} is too far (distance {min_distance/1000:.2f} km > {threshold/1000:.2f} km). Skipping.")
            closest_country['merged'] = True  # Marquer comme traité (mais non fusionné)
            continue

        logging.info(f"Merging country: {closest_country['name']} (distance {min_distance/1000:.2f} km)")

        # Fusionner le pays avec le réseau fusionné
        merged_network = merge_two_networks(merged_network, closest_country, distance_threshold=threshold)

        # Marquer le pays comme fusionné
        closest_country['merged'] = True

    # Préparer les données pour l'exportation
    merged_nodes = merged_network['nodes']
    merged_edges = merged_network['edges']

    # Traitement final
    merged_edges = addKm(merged_edges, 4087)

    merged_nodes = merged_nodes.drop(columns=["osmid", "id"], errors='ignore')
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

    # Gérer les pays non fusionnés (îles ou autres continents)
    skipped_countries = [cn for cn in country_networks if not cn['merged']]
    for cn in skipped_countries:
        logging.info(f"Skipped country: {cn['name']} (considered as island or on another continent)")
