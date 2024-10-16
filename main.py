#How to use netsimp
#Example done with Turkmenistan and Uzbekistan
#Download files here:
#https://download.geofabrik.de/asia.html

from extract_OSM import extract_from_PBF
from simplification_A import simplification_A
from simplification_B import simplification_B
from merging_networks import merge_networks
import logging

logging.basicConfig(level=logging.INFO)

input_PBF = "C:\\Users\\gay\\Documents\\netsimp\\test\\PBF_files" #The only line to change to make the process
                                                                    # data works
extract_from_PBF(input_folder=input_PBF)

simplification_A(input_PBF)

simplification_B(input_PBF)

merge_networks(input_PBF)
