from .ppo import PPOTrainerWrapper
from .grpo import GRPOTrainer
from .dapo import DAPOTrainer
from .rlvr import RLVRTrainer

TRAINERS: dict[str, type] = {
    "ppo": PPOTrainerWrapper,
    "grpo": GRPOTrainer,
    "dapo": DAPOTrainer,
    "rlvr": RLVRTrainer,
}
