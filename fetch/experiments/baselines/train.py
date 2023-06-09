import os
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        #Args.debug = True
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                Args.n_workers = [2, 8, 16, 20]
                Args.n_epochs = [2,2,2,2]#[50, 150, 200, 500]
            Args.seed = [100, 200, 300,]

    for i, deps in sweep.items():
        save_path = f"{os.getcwd()}/results/baselines/{deps['Args.env_name']}/{deps['Args.seed']}"
        os.environ["ML_LOGGER_ROOT"] = save_path
        main(deps)
