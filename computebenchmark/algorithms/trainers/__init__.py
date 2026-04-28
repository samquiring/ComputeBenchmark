import importlib

_REGISTRY = {
    "ppo": ("ppo", "PPOTrainerWrapper"),
    "grpo": ("grpo", "GRPOTrainerWrapper"),
    "dapo": ("dapo", "DAPOTrainer"),
    "rlvr": ("rlvr", "RLVRTrainer"),
}

TRAINERS = list(_REGISTRY.keys())


def get_trainer(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Choose from: {', '.join(TRAINERS)}")
    module_name, class_name = _REGISTRY[name]
    module = importlib.import_module(f".{module_name}", package=__name__)
    return getattr(module, class_name)
