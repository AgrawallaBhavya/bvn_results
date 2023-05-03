import os
from params_proto.neo_hyper import Sweep
def get_dir(ff):
    if ff is not None:
        return ff
    return "none"


if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.critic_type = 'state_asym_metric'
        Args.critic_loss_type = 'td'

        Args.env_name = 'FetchPickAndPlace-v1'
        Args.n_workers = 16
        Args.n_epochs = 200
        Args.seed = 300
        Args.metric_embed_dim = 16
        Args.fourier_features = None

    for i, deps in sweep.items():
        os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/{get_dir(deps['Args.fourier_features'])}/{deps['Args.env_name']}/{deps['Args.seed']}"
        main(deps)
