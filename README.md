# OSMsimp: Adaptive Workflow to Prepare and Simplify Transport Networks from OpenStreetMap

## Overview

This repository contains the implementation of a comprehensive workflow for extracting and simplifying transport networks from **OpenStreetMap** (OSM) data. The goal of this workflow is to streamline large-scale transportation networks, making them easier to analyze and model. The workflow utilizes Python libraries such as **OSMnx** and **esy-filter**, in addition to custom functions designed specifically for simplifying real-world transportation networks. The workflow can extract various types of transport infrastructures such as roads, railways, and pipelines at any desired scale.

## Authors

- **Adrien Fauste-Gay**  
  Department of Physics, École Normale Supérieure, Paris, France  
  [adrien.gay@ens.psl.eu](mailto:adrien.gay@ens.psl.eu)

- **Célian Colon**  
  Supervisor, Exploratory Modeling Research Team, IIASA, Laxenburg, Austria  
  [celian.colon@polytechnique.org](mailto:celian.colon@polytechnique.org)

## Keywords

- OpenStreetMap
- OSMnx
- Network Simplification
- Transport Networks

## Abstract

This repository implements a workflow that extracts and simplifies transport networks from **OpenStreetMap** data. The workflow is designed to remove unnecessary nodes and edges while preserving the essential characteristics of the base network, enabling simplified large-scale network analysis. We combine existing libraries such as **OSMnx** and **esy-filter** to handle raw **PBF** files, reducing network complexity for efficient analysis and modeling. The resulting simplified networks retain essential information but eliminate artifacts and redundant data, making them suitable for macro-economic models and infrastructure analysis.

## Features

- **Transport Network Extraction**: Efficient extraction of various transport infrastructure, including roads, railways, and pipelines, from OSM data.
- **Network Simplification**: Removal of unnecessary nodes, edges, and redundant intersections while maintaining the network's backbone.
- **Large-Scale Analysis**: Suitable for large-scale analysis and modeling of transportation networks at the national and international level.
- **Merging of Network Files**: Merge multiple simplified networks into a unified large-scale transport network.

## Workflow Description

### Data Extraction

The workflow utilizes **PBF** files downloaded from [Geofabrik](https://download.geofabrik.de/) to extract relevant transport infrastructure data from OpenStreetMap. We use the **esy-osmfilter** library to filter out unnecessary data and extract the main road types labeled as `motorway`, `motorway_link`, `trunk`, and `trunk_link`. The extracted network data is saved in **GeoJSON** format, containing the nodes and edges of the transport network.

### Network Simplification

The network simplification is divided into two phases:

- **Simplification Phase A**: Involves removing unnecessary road intersections and degree-2 nodes using **OSMnx**'s functions such as `consolidate_intersections` and `simplify_graph`. This phase simplifies intersections and cleans up redundant details while preserving the overall structure of the network.

- **Simplification Phase B**: Further reduces complexity by removing non-network nodes, remaining degree-2 nodes, and using **DBSCAN** clustering to merge nearby nodes into a simplified representation. This phase results in a clean, connected network structure, with only the main transport paths preserved.

### Merging Networks

When working with large regions that require multiple **PBF** files, the workflow merges individual network segments into a unified network. We use custom merging functions to connect close nodes across different network files and update the edge lengths accordingly. The final output is a single network suitable for analysis.

## Example: Uzbekistan and Turkmenistan

We applied the workflow to the road networks of **Uzbekistan** and **Turkmenistan**. Initially, the combined **PBF** files for these countries contained over 13 million nodes and 1.8 million edges. After applying the simplification workflow, the final output consists of only 97 nodes and 122 edges, representing just 0.0007% of the original dataset. This drastic reduction in complexity retains the essential transportation paths while eliminating unnecessary details, making the network easier to analyze and model.

## Data Requirements

To run the workflow, you need:

- **OpenStreetMap PBF files**: Downloaded from [Geofabrik](https://download.geofabrik.de/).
- **Python packages**:
  - `OSMnx`
  - `esy-osmfilter`
  - `NetworkX`
  - `GeoPandas`
  - `DBSCAN`

## Installation

Clone the repository and install the necessary Python packages:

```bash
git clone https://github.com/AdrienFausteGay/osmsimp.git
cd osmsimp
pip install -r requirements.txt
