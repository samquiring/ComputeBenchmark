from .grpo import GRPOTrainer


class RLVRTrainer(GRPOTrainer):
    """Reinforcement Learning from Verifiable Rewards.

    Architecturally identical to GRPO; the distinction is conceptual: no neural
    reward model is used at any stage. Reward is strictly binary (1 for correct
    numerical answer, 0 otherwise). This is the default in GRPOTrainer but made
    explicit here so the paper can discuss it as a separate ablation condition.
    """

    def _compute_reward(self, response: str, ground_truth: str) -> float:
        from ...data.gsm8k import extract_answer
        predicted = extract_answer(response)
        # Strict match only — no format bonuses, no partial credit
        return 1.0 if predicted is not None and predicted == ground_truth else 0.0
