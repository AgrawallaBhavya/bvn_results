import os
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.critic_type = 'state_asym_metric'
        Args.critic_loss_type = 'td'

        Args.env_name = <ENV_NAME>
        Args.n_workers = <N_WORKERS>
        Args.n_epochs = <N_EPOCHS>
        Args.seed = <SEED>
        Args.metric_embed_dim = <METRIC_EMBED_DIM>
        Args.use_critic_fourier_features = <USE_CRITIC_FOURIER_FEATURES>
        #Args.state_fourier_features = <STATE_FOURIER_FEATURES>
        #Args.action_fourier_features = <ACTION_FOURIER_FEATURES>
        #Args.goal_fourier_features = <GOAL_FOURIER_FEATURES>
        #Args.b_scale = <B_SCALE>

    for i, deps in sweep.items():
        #os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/" + str(deps['Args.use_critic_fourier_features']) + f"/{deps['Args.env_name']}/{deps['Args.state_fourier_features']}/{deps['Args.b_scale']}/{deps['Args.seed']}"
        os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/" + str(deps['Args.use_critic_fourier_features']) + f"-512-share_state/{deps['Args.env_name']}/{deps['Args.seed']}"
        main(deps)




