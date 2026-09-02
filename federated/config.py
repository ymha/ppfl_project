import os

import tomli

# flwr's own `flwr run` CLI requires a running SuperLink control-plane
# process even for local "simulation" federations in this flwr version, and
# the lighter-weight flwr.simulation.run_simulation() Python API (what
# run_local_simulation.py actually uses) does NOT auto-populate
# Context.run_config from pyproject.toml the way the CLI path would (it
# starts every run with an empty run_config unless the caller supplies one
# via a private API). So server_app.py and client_app.py read the
# `[tool.flwr.app.config]` table directly instead of relying on
# Context.run_config.
PYPROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")


def load_run_config():
    with open(PYPROJECT_PATH, "rb") as f:
        data = tomli.load(f)
    return data["tool"]["flwr"]["app"]["config"]
