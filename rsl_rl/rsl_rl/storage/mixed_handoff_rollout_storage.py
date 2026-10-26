"""Fixed rollout tensors with explicit action ownership and segment boundaries."""
import torch


class MixedHandoffRolloutStorage:
    def __init__(self, num_envs, num_steps, actor_obs=738, critic_obs=126,
                 actions=29, device="cpu"):
        shapes = dict(observations=actor_obs, next_observations=actor_obs,
                      critic_observations=critic_obs, next_critic_observations=critic_obs,
                      actions=actions, mu=actions, sigma=actions, amp_observations=540)
        self.data = {k: torch.zeros(num_steps, num_envs, n, device=device)
                     for k, n in shapes.items()}
        for k in ("rewards", "values", "next_values", "log_prob", "returns", "advantages"):
            self.data[k] = torch.zeros(num_steps, num_envs, device=device)
        for k in ("student_valid", "amp_valid", "dones", "timeouts", "failures", "next_student"):
            self.data[k] = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        self.step = 0
        self.num_steps = num_steps

    def add(self, **transition):
        if self.step >= self.num_steps:
            raise RuntimeError("Mixed handoff rollout overflow")
        if set(transition) != set(self.data) - {"returns", "advantages"}:
            raise ValueError("Incomplete mixed handoff transition")
        if (transition["amp_valid"] & ~transition["student_valid"]).any():
            raise ValueError("AMP-valid samples must belong to Student")
        for k, v in transition.items():
            self.data[k][self.step].copy_(v)
        self.step += 1

    def compute_returns(self, gamma, lam):
        d = self.data
        valid = d["student_valid"][:self.step]
        advantage = torch.zeros_like(d["values"][0])
        d["advantages"].zero_()
        d["returns"].zero_()
        for t in reversed(range(self.step)):
            continuing = valid[t] & ~d["dones"][t] & ~d["failures"][t] & d["next_student"][t]
            bootstrap = valid[t] & ~d["failures"][t] & (
                continuing | (d["dones"][t] & d["timeouts"][t]))
            # torch.where also prevents poisoned Teacher placeholders leaking
            # through 0 * NaN in this recursion.
            delta = (d["rewards"][t] + gamma * torch.where(bootstrap, d["next_values"][t], 0.)
                     - d["values"][t])
            trace = continuing
            if t + 1 < self.step:
                trace = trace & valid[t + 1]
            advantage = torch.where(valid[t], delta + gamma * lam * torch.where(trace, advantage, 0.), 0.)
            d["advantages"][t] = advantage
            d["returns"][t] = torch.where(valid[t], advantage + d["values"][t], 0.)
        selected = d["advantages"][:self.step][valid]
        self.advantage_mean = selected.mean().item() if selected.numel() else 0.
        self.advantage_std = selected.std(unbiased=False).item() if selected.numel() else 0.
        if selected.numel() >= 2:
            d["advantages"][:self.step][valid] = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)

    def batches(self, num_mini_batches, num_epochs):
        valid = self.data["student_valid"][:self.step].flatten().nonzero().flatten()
        if not valid.numel():
            return
        flat = {k: v[:self.step].flatten(0, 1) for k, v in self.data.items()}
        for _ in range(num_epochs):
            order = valid[torch.randperm(valid.numel(), device=valid.device)]
            for ids in order.tensor_split(min(num_mini_batches, valid.numel())):
                yield {k: v[ids] for k, v in flat.items()}

    def amp_samples(self):
        return self.data["amp_observations"][:self.step][self.data["amp_valid"][:self.step]]

    def clear(self):
        self.step = 0
