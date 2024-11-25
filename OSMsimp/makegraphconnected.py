import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from sklearn.neighbors import BallTree
import numpy as np
from geopy.distance import geodesic
import logging
logging.basicConfig(level=logging.INFO)

def make_graph_connected(nodes_df, edges_df, threshold = None):
    """
    Assure que le graphe défini par nodes_df et edges_df est connexe en ajoutant des arêtes
    entre les nœuds les plus proches des composantes déconnectées, sous réserve d'une distance maximale.

    Paramètres:
    - nodes_df: GeoDataFrame des nœuds, avec 'osmid' comme identifiant de nœud.
    - edges_df: GeoDataFrame des arêtes, avec 'u' et 'v' comme identifiants des nœuds d'extrémité.
    - threshold: Distance maximale pour ajouter une arête (None pour ignorer la limite).

    Retourne:
    - nodes_df: GeoDataFrame des nœuds (inchangé).
    - edges_df: GeoDataFrame des arêtes mis à jour avec les arêtes supplémentaires.
    """
    # Assurer que 'osmid' est l'identifiant des nœuds
    nodes_df = nodes_df.set_index('osmid', drop=False)
    
    # Construire le graphe non orienté avec NetworkX
    G = nx.Graph()
    # Ajouter les nœuds
    for idx, row in nodes_df.iterrows():
        G.add_node(row['osmid'])
    # Ajouter les arêtes
    for idx, row in edges_df.iterrows():
        u = row['u']
        v = row['v']
        G.add_edge(u, v)
    
    # Trouver les composantes connexes
    components = list(nx.connected_components(G))
    num_components = len(components)
    logging.debug(f"Nombre de composantes connexes : {num_components}")
    
    if num_components <= 1:
        logging.debug("Le graphe est déjà connexe.")
        return nodes_df.reset_index(drop=True), edges_df.reset_index(drop=True)
    
    else:
        # Préparer une liste pour les nouvelles arêtes
        new_edges = []
        new_edges_data = []
        
        # Convertir la liste des composantes en une liste de GeoDataFrames
        components_nodes = []
        for comp in components:
            comp_nodes = nodes_df.loc[list(comp)]
            components_nodes.append(comp_nodes)
        
        # # Construire un arbre pour chaque composante
        # trees = []
        # for comp_nodes in components_nodes:
        #     # Extraire les coordonnées des nœuds
        #     coords = np.array([(geom.x, geom.y) for geom in comp_nodes.geometry])
        #     # Construire un BallTree pour la composante
        #     tree = BallTree(coords, metric='euclidean')
        #     trees.append((tree, comp_nodes))
        
        # Connecter les composantes
        # On va créer un arbre minimal couvrant entre les composantes pour minimiser les distances
        # Créer une matrice pour stocker les distances minimales entre composantes
        comp_indices = range(len(components_nodes))
        connected = set()
        # On va utiliser Kruskal pour connecter les composantes
        edges_to_add = []
        
        # Calculer les distances entre les composantes
        from itertools import combinations
        comp_pairs = list(combinations(comp_indices, 2))
        distances = []
        pairs = []
        # for i, j in comp_pairs:
        #     tree_i, nodes_i = trees[i]
        #     tree_j, nodes_j = trees[j]
        #     # Trouver les paires de nœuds les plus proches entre les deux composantes
        #     coords_i = np.array([(geom.x, geom.y) for geom in nodes_i.geometry])
        #     coords_j = np.array([(geom.x, geom.y) for geom in nodes_j.geometry])
        #     # Calculer les distances minimales entre tous les points
        #     dists, idxs = tree_i.query(coords_j, k=1)
        #     min_dist_idx = np.argmin(dists)
        #     min_dist = dists[min_dist_idx][0]
        #     idx_i = idxs[min_dist_idx][0]
        #     idx_j = min_dist_idx
        #     node_i = nodes_i.iloc[idx_i]['osmid']
        #     node_j = nodes_j.iloc[idx_j]['osmid']
        #     if threshold is None or min_dist <= threshold:
        #         pairs.append((node_i, node_j, min_dist))
        #         distances.append(min_dist)

        # Calculer les distances géodésiques entre les composantes
        for i, j in comp_pairs:
            nodes_i = components_nodes[i]
            nodes_j = components_nodes[j]
            
            # Initialiser les variables pour stocker la distance minimale et les nœuds correspondants
            min_dist = float('inf')
            node_i_closest = None
            node_j_closest = None
            
            # Parcourir tous les nœuds des deux composantes
            for idx_i, row_i in nodes_i.iterrows():
                coord_i = (row_i.geometry.y, row_i.geometry.x)
                for idx_j, row_j in nodes_j.iterrows():
                    coord_j = (row_j.geometry.y, row_j.geometry.x)
                    
                    # Calculer la distance géodésique
                    dist = geodesic(coord_i, coord_j).meters  # Conversion en mètres
                    
                    # Mettre à jour si une distance plus petite est trouvée
                    if dist < min_dist:
                        min_dist = dist
                        node_i_closest = row_i['osmid']
                        node_j_closest = row_j['osmid']
            
            # Ajouter la paire si elle respecte le seuil
            if threshold is None or min_dist <= threshold:
                pairs.append((node_i_closest, node_j_closest, min_dist))
                distances.append(min_dist)

        if len(distances)==0:
            logging.debug("0 arête ajoutée. Les composantes sont trop loin")
            return nodes_df.reset_index(drop=True), edges_df.reset_index(drop=True)
        
        # Trier les paires par distance croissante
        pairs = sorted(pairs, key=lambda x: x[2])
        
        # Utiliser l'algorithme de Kruskal pour ajouter les arêtes nécessaires
        uf = UnionFind(len(components_nodes))
        for node_i, node_j, dist in pairs:
            comp_i = find_component_index(node_i, components_nodes)
            comp_j = find_component_index(node_j, components_nodes)
            if uf.find(comp_i) != uf.find(comp_j):
                # Ajouter l'arête
                new_edge = {'u': node_i, 'v': node_j, 'geometry': LineString([
                    nodes_df.loc[node_i].geometry, nodes_df.loc[node_j].geometry
                ])}
                new_edges_data.append(new_edge)
                # Union des composantes
                uf.union(comp_i, comp_j)
            if uf.num_sets == 1:
                break  # Le graphe est maintenant connexe
        
        # Ajouter les nouvelles arêtes au GeoDataFrame des arêtes
        new_edges_df = gpd.GeoDataFrame(new_edges_data, crs=edges_df.crs)
        edges_df = pd.concat([edges_df, new_edges_df], ignore_index=True)
        
        logging.debug(f"{len(new_edges_data)} arête(s) ajoutée(s) pour rendre le graphe connexe.")
        return nodes_df.reset_index(drop=True), edges_df.reset_index(drop=True)

# Classe Union-Find pour l'algorithme de Kruskal
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.num_sets = n
    
    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]  # Compression de chemin
            u = self.parent[u]
        return u
    
    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root == v_root:
            return
        if self.rank[u_root] < self.rank[v_root]:
            self.parent[u_root] = v_root
        else:
            self.parent[v_root] = u_root
            if self.rank[u_root] == self.rank[v_root]:
                self.rank[u_root] += 1
        self.num_sets -= 1

def find_component_index(node_osmid, components_nodes):
    for idx, comp_nodes in enumerate(components_nodes):
        if node_osmid in comp_nodes['osmid'].values:
            return idx
    return -1  # Erreur si non trouvé (ne devrait pas arriver)
