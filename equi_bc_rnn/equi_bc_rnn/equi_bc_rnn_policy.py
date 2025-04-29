from robomimic.models.base_nets import MLP
from robomimic.models.obs_nets import ObservationGroupEncoder
from robomimic.models.policy_nets import RNNActorNetwork

def EquivariantBCRNNPolicy(obs_shapes, ac_dim, algo_config, obs_config, goal_shapes=None):
    obs_encoder = ObservationGroupEncoder(
        input_obs_group_shapes=obs_shapes,
        output_shape=(512,),
        encoder_kwargs=obs_config.encoder,
    )
    return RNNActorNetwork(
        action_dim=ac_dim,
        obs_group_encoder=obs_encoder,
        goal_shapes=goal_shapes,
        per_step_mlp_dims=[512],
        rnn_horizon=algo_config.rnn.horizon,
        rnn_hidden_dim=algo_config.rnn.hidden_dim,
        rnn_type=algo_config.rnn.rnn_type,
        rnn_num_layers=algo_config.rnn.num_layers,
    )
