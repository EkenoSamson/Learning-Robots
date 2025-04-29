from robomimic.config.base_config import BaseConfig

class EquivariantBCRNNConfig(BaseConfig):
    ALGO_NAME = "equi_bc_rnn"

    def algo_config(self):
        self.algo.rnn.enabled = True
        self.algo.rnn.horizon = 15
        self.algo.rnn.hidden_dim = 512
        self.algo.rnn.num_layers = 2
        self.algo.rnn.rnn_type = "LSTM"
        self.algo.rnn.open_loop = False
        self.algo.rnn.kwargs = {"bidirectional": False}

        # Equivariant encoder options
        self.algo.encoder.use_equivariance = True
        self.algo.encoder.layer_dims = [256, 256]
        self.algo.encoder.equiv_type = "SO3"  # or SE3, based on your method

        # Optimization
        self.algo.optim_params.policy.learning_rate.initial = 5e-4
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.regularization.L2 = 0.0

        # Loss
        self.algo.loss.l2_weight = 1.0
        self.algo.loss.l1_weight = 0.0
        self.algo.loss.cos_weight = 0.0

        # GMM parameters (not used but required by config JSON)
        self.algo.gmm.enabled = False
        self.algo.gmm.num_modes = 5
        self.algo.gmm.min_std = 0.0001
        self.algo.gmm.std_activation = "softplus"
        self.algo.gmm.low_noise_eval = True

        self.algo.optim_params.policy.learning_rate.initial = 0.0005
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep"  # <- REQUIRED

        # Optional L2 regularization
        self.algo.optim_params.policy.regularization.L2 = 0.0
