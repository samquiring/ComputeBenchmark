"""RLVR: Reinforcement Learning from Verifiable Rewards.

Architecturally identical to GRPO. The distinction is explicit: reward is
strictly binary (1 = exact numerical match, 0 = otherwise). No format bonuses,
no partial credit, no learned reward model at any stage.
"""

from .grpo import GRPOTrainerWrapper


class RLVRTrainer(GRPOTrainerWrapper):
    """GRPO with an unambiguous binary verifier reward.

    Listed separately so the paper can discuss the reward-signal design choice
    as an independent axis from the optimization algorithm (GRPO).
    """

    def _reward_funcs(self) -> list:
        from ...data.gsm8k import extract_answer

        def binary_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
            return [
                1.0 if extract_answer(c) == a else 0.0
                for c, a in zip(completions, answer)
            ]

        return [binary_reward]
