import os
# os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True" #Ligne ajoutée et non présente sur l'ancien git

from esy.osmfilter import run_filter
from esy.osmfilter import Node, Way, Relation
from esy.osmfilter import export_geojson

import shutil
import logging

prefilter_dict = {1: {      Node: {},
                            Way: {"highway": ["motorway", "motorway_link", "trunk", "trunk_link"]},
                            Relation: {}
                        },
                   2: {
                        Node: {},
                        Way: {
                            "highway": [
                                "motorway", "motorway_link",
                                "trunk", "trunk_link",
                                "primary", "primary_link",
                                "secondary", "secondary_link"
                            ]
                        },
                        Relation: {}
                    },
                    3: {
                        Node: {},
                        Way: {
                            "highway": [
                                "motorway", "motorway_link",
                                "trunk", "trunk_link",
                                "primary", "primary_link",
                                "secondary", "secondary_link",
                                "tertiary", "tertiary_link"
                            ]
                        },
                        Relation: {}
                    }
    }

whitefilter_dict = {1: [
                            [("highway", "motorway")],
                            [("highway", "motorway_link")],
                            [("highway", "trunk")],
                            [("highway", "trunk_link")]
                        ],
                    2: [
                        [("highway", "motorway")],
                        [("highway", "motorway_link")],
                        [("highway", "trunk")],
                        [("highway", "trunk_link")],
                        [("highway", "primary")],
                        [("highway", "primary_link")],
                        [("highway", "secondary")],
                        [("highway", "secondary_link")]
                    ],
                    3: [
                        [("highway", "motorway")],
                        [("highway", "motorway_link")],
                        [("highway", "trunk")],
                        [("highway", "trunk_link")],
                        [("highway", "primary")],
                        [("highway", "primary_link")],
                        [("highway", "secondary")],
                        [("highway", "secondary_link")],
                        [("highway", "tertiary")],
                        [("highway", "tertiary_link")]
                    ]
    }

blackfilter_dict = {1: [("highway", "primary"), ("highway", "secondary")],
                    2: [
                        ("highway", "tertiary"),
                        ("highway", "tertiary_link"),
                        ("highway", "unclassified"),
                        ("highway", "residential"),
                        ("highway", "service"),
                        ("highway", "living_street"),
                        ("highway", "pedestrian"),
                        ("highway", "track"),
                        ("highway", "road"),
                        ("highway", "path"),
                        ("highway", "footway"),
                        ("highway", "cycleway"),
                        ("highway", "steps"),
                        ("highway", "construction"),
                        ("highway", "bus_guideway"),
                        ("highway", "escape"),
                        ("highway", "raceway"),
                        ("highway", "bridleway"),
                        ("highway", "corridor"),
                        ("highway", "proposed"),
                        ("highway", "motorway_junction"),
                        ("highway", "platform")
                    ],
                    3: [
                        ("highway", "unclassified"),
                        ("highway", "residential"),
                        ("highway", "service"),
                        ("highway", "living_street"),
                        ("highway", "pedestrian"),
                        ("highway", "track"),
                        ("highway", "road"),
                        ("highway", "path"),
                        ("highway", "footway"),
                        ("highway", "cycleway"),
                        ("highway", "steps"),
                        ("highway", "construction"),
                        ("highway", "bus_guideway"),
                        ("highway", "escape"),
                        ("highway", "raceway"),
                        ("highway", "bridleway"),
                        ("highway", "corridor"),
                        ("highway", "proposed"),
                        ("highway", "motorway_junction"),
                        ("highway", "platform")
                    ]
    }

def extract_from_PBF(input_folder, level=2):
    logging.info("EXTRACT DATA FROM PBF FILES")
    print("EXTRACT DATA FROM PBF FILES")
    prefilter = prefilter_dict[level]
    whitefilter = whitefilter_dict[level]
    blackfilter = blackfilter_dict[level]
    for file in [file for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file))]:
        logging.info("Processed file: " + str(file))
        print("Processed file: " + str(file))
        PBF_inputfile = os.path.join(input_folder, file)
        JSON_path = os.path.join(input_folder, "JSON")
        if not os.path.exists(JSON_path):
            os.mkdir(JSON_path)
        JSON_outputfile = os.path.join(JSON_path, #'C:/Users', 'gay', 'Documents', 'Github', 'travail perso', "json",
                                       file[:-15] + '.json')
        logging.info("Extracting network from:" + str(file))
        print("Extracting network from:" + str(file))
        [Data, Elements] = run_filter('noname',
                                      PBF_inputfile,
                                      JSON_outputfile,
                                      prefilter,
                                      whitefilter,
                                      blackfilter,
                                      NewPreFilterData=True,
                                      CreateElements=True,
                                      LoadElements=False,
                                      verbose=False,
                                      multiprocess=False)

        Elements["noname"]["Node"] = Data["Node"]

        output_folder = os.path.join(os.path.dirname(input_folder), "raw_files")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        export_geojson(Elements['noname']['Way'], Data,
                       filename=os.path.join(output_folder, file[:-15] + '_edges.geojson'),
                       jsontype='Line')
        export_geojson(Elements['noname']['Node'], Data,
                       filename=os.path.join(output_folder, file[:-15] + '_nodes.geojson'),
                       jsontype='Point')

        if not os.path.exists(os.path.join(input_folder, "Done")):
            os.makedirs(os.path.join(input_folder, "Done"))
        shutil.move(os.path.join(input_folder, file), os.path.join(input_folder, "Done", file))
