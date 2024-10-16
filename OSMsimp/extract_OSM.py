import os
# os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True" #Ligne ajoutée et non présente sur l'ancien git

from esy.osmfilter import run_filter
from esy.osmfilter import Node, Way, Relation
from esy.osmfilter import export_geojson

import shutil
import logging


def extract_from_PBF(input_folder, prefilter=None,
                     whitefilter=None, blackfilter=None):
    logging.info("EXTRACT DATA FROM PBF FILES")
    print("EXTRACT DATA FROM PBF FILES")
    if prefilter is None:
        prefilter = {Node: {}, Way: {
            "highway": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"], }, Relation: {}}
    if whitefilter is None:
        whitefilter = [(("highway", "trunk"),), (("highway", "trunk_link"),), (("highway", "motorway"),),
                       (("highway", "motorway_link"),)]
    if blackfilter is None:
        blackfilter = [("highway", "secondary")]
    for file in [file for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file))]:
        logging.info("Processed file: " + str(file))
        print("Processed file: " + str(file))
        PBF_inputfile = os.path.join(input_folder, file)
        JSON_outputfile = os.path.join('C:/Users', 'gay', 'Documents', 'Github', 'travail perso', "json",
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
