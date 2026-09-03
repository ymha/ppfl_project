import argparse

from flwr.simulation import run_simulation

from federated.client_app import app as client_app
from federated.config import load_run_config
from federated.server_app import app as server_app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=4, help="CPUs reserved per client activation")
    # 1.0 = the whole GPU per client activation, which is what forces Ray to
    # run client activations one at a time on this single-GPU box (see
    # server_app.py/client_app.py's BaseModelCache risk notes).
    parser.add_argument("--num-gpus", type=float, default=1.0, help="GPU share reserved per client activation")
    return parser.parse_args()


def main():
    args = parse_args()
    run_config = load_run_config()
    num_clients = int(run_config["num-clients"])

    backend_config = {
        "client_resources": {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    main()
