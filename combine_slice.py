"""
Combine multiple slices into one 3D *.nii.gz stacks
"""

import os
import sys

# import user defined library
from Util.parse_config import parse_config
from Util.preprocess_lib import combine_slices


def main(config_file):
    config = parse_config(config_file)
    config = config["para"]
    # for key in config:
    #     print(type(key), type(config[key]))
    combine_slices(config)


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise Exception("Invaid number of input parameters!")
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))  # make sure config_file is a file name
    main(config_file)