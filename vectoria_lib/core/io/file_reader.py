#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import json
import os
import sys


def json_reader(filepath):
    """Read and parse a JSON file."""
    filecontained = "-999"
    try:
        if os.path.isfile(filepath):
            with open(filepath, encoding="utf-8") as f:
                filecontained = json.load(f)
        else:
            sys.exit()
    except Exception:
        sys.exit("ERROR JSON file not readable " + str(filepath))
    return filecontained


def filetostring(filepath):
    onestring = ""
    try:
        multistringfile = generic_file_reader(filepath)
        for r in multistringfile:
            onestring = onestring + r
    except Exception:
        print("ERROR file doesn't exist: " + filepath)
    return onestring


def generic_file_reader(filename):
    """Read a generic ASCII file and return a list of lines."""
    rows_from_file = []
    if os.path.isfile(filename):
        with open(filename, encoding="utf-8") as file_toread:
            rows_from_file = file_toread.readlines()
    else:
        print("ERROR file doesn't exist: " + filename)
    return rows_from_file
