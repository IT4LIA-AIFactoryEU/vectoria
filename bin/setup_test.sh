#!/bin/bash
#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

# Get the virtual environment path
venv_path=$VIRTUAL_ENV

# Get the Python version (major.minor)
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo $python_version

# Construct the site-packages path
lib_path="$venv_path/lib/python$python_version/site-packages"
echo $lib_path

# Add lib_path to PYTHONPATH if not already present
if [[ ":$PYTHONPATH:" != *":$lib_path:"* ]]; then
  export PYTHONPATH="$lib_path${PYTHONPATH:+:$PYTHONPATH}"
fi

# Print the updated PYTHONPATH
echo "$PYTHONPATH"
