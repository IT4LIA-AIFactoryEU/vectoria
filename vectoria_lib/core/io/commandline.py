import argparse
import os
import sys
from pathlib import Path

from vectoria_lib.core.io import file_reader
from vectoria_lib.core.paths import ETC_DIR


def create_arg_dict_from_string(a):
    """Transform string values in an arg dict into Python callable objects."""
    for k, v in a.items():
        if k == "required":
            a[k] = True if v == "True" else False
        if k == "type":
            if v == "str":
                a[k] = str
            elif v == "int":
                a[k] = int
            elif v == "float":
                a[k] = float
            elif v == "open":
                a[k] = open


def create_cl_parser_from_json(parser, json_args: str | Path) -> argparse.ArgumentParser:
    """Build an argparse subparser from a JSON config file."""
    check_setting_path = get_setting_file_path(json_args)
    cl_setting = file_reader.json_reader(check_setting_path[0])
    for arg in cl_setting["flag"]:
        create_arg_dict_from_string(arg["param"])
        parser.add_argument(arg["name"], **arg["param"])
    return parser


def get_setting_file_path(filename) -> list:
    """Return paths to CLI setting files found under etc/custom or etc/shared."""
    check_setting_path = []
    check_setting_find = True

    check_setting_path_default = ETC_DIR / "shared" / "cli" / filename
    check_setting_path_custom = ETC_DIR / "custom" / "cli" / filename

    if os.path.exists(check_setting_path_default):
        check_setting_path.append(check_setting_path_default)
        check_setting_find = False

    if os.path.exists(check_setting_path_custom):
        check_setting_path.append(check_setting_path_custom)
        check_setting_find = False

    if check_setting_find:
        sys.exit("ERROR: CLI setting file not found: " + str(filename))

    return check_setting_path


def cl_convert_to_dict(args):
    """Convert an argparse Namespace to a dict, dropping None/falsy values."""
    return {k: v for k, v in vars(args).items() if v is not None}
