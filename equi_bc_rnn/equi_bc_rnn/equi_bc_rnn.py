from collections import OrderedDict
import torch.nn as nn
from robomimic.algo import register_algo_factory_func
from robomimic.algo.algo import PolicyAlgo
from robomimic.models.policy_nets import RNNActorNetwork
from robomimic.models.obs_nets import ObservationGroupEncoder
from robomimic.utils.train_utils import TorchUtils

class EquivariantBCRNN(PolicyAlgo):
    def _create_networks(self):
        self.nets = nn.ModuleDict()
        obs_encoder = ObservationGroupEncoder(
            input_obs_group_shapes=self.obs_shapes,
            output_shape=(512,),
            encoder_kwargs=self.obs_encoder_kwargs,
        )
        rnn_net = RNNActorNetwork(
            action_dim=self.ac_dim,
            rnn_hidden_dim=self.algo_config.rnn.hidden_dim,
            rnn_horizon=self.algo_config.rnn.horizon,
            rnn_type=self.algo_config.rnn.rnn_type,
            rnn_num_layers=self.algo_config.rnn.num_layers,
            obs_group_encoder=obs_encoder,
            goal_shapes=self.goal_shapes,
            per_step_mlp_dims=[512],
        )
        self.nets["policy"] = rnn_net
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        return super().train_on_batch(batch, epoch, validate=validate)

    def get_action(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        return self.nets["policy"](obs_dict=obs_dict, goal_dict=goal_dict)

@register_algo_factory_func("equi_bc_rnn")
def algo_config_to_class(algo_config):
    return EquivariantBCRNN, {}
