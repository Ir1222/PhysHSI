"""Student-only PPO, following HIMPPO's clipping, KL and smoothness objectives."""
import torch
from torch import nn


class MixedHandoffPPO:
    def __init__(self, actor_critic, amp=None, amp_normalizer=None, motion_buffer=None,
                 num_learning_epochs=5, num_mini_batches=4, clip_param=.2, gamma=.99,
                 lam=.95, value_loss_coef=1., entropy_coef=.01, learning_rate=1e-3,
                 max_grad_norm=1., use_clipped_value_loss=True, schedule="adaptive",
                 desired_kl=.01, value_smoothness_coef=.1, smoothness_upper_bound=1.,
                 smoothness_lower_bound=.1, amp_batch_size=256):
        self.actor_critic, self.amp = actor_critic, amp
        self.amp_normalizer, self.motion_buffer = amp_normalizer, motion_buffer
        self.num_learning_epochs, self.num_mini_batches = num_learning_epochs, num_mini_batches
        self.clip_param, self.gamma, self.lam = clip_param, gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.learning_rate, self.max_grad_norm = learning_rate, max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule, self.desired_kl = schedule, desired_kl
        self.value_smoothness_coef = value_smoothness_coef
        self.smoothness_upper_bound, self.smoothness_lower_bound = smoothness_upper_bound, smoothness_lower_bound
        self.amp_batch_size = amp_batch_size
        if num_learning_epochs < 1 or num_mini_batches < 1 or amp_batch_size < 1:
            raise ValueError("PPO epochs, minibatches and AMP batch size must be positive")
        if not 0 <= smoothness_lower_bound < smoothness_upper_bound:
            raise ValueError("Smoothness bounds require 0 <= lower < upper")
        groups = [dict(params=actor_critic.parameters(), name="student")]
        if amp is not None:
            groups += [dict(params=amp.trunk.parameters(), weight_decay=1e-3, name="amp_trunk"),
                       dict(params=amp.amp_linear.parameters(), weight_decay=.1, name="amp_head")]
        self.optimizer = torch.optim.Adam(groups, lr=learning_rate)
        self.optimizer_updates = 0  # nonempty rollout updates
        self.optimizer_steps = 0  # actual Adam steps

    def ppo_loss(self, b):
        ac = self.actor_critic
        ac.update_distribution(b["observations"])
        log_prob = ac.get_actions_log_prob(b["actions"])
        mu, sigma, entropy = ac.action_mean, ac.action_std, ac.entropy.mean()
        value = ac.evaluate(b["critic_observations"]).squeeze(-1)
        with torch.no_grad():
            kl = (torch.log(sigma / b["sigma"] + 1e-5) +
                  (b["sigma"].square() + (b["mu"] - mu).square()) / (2 * sigma.square()) - .5).sum(-1).mean()
        ratio = (log_prob - b["log_prob"]).exp()
        surrogate = torch.maximum(-b["advantages"] * ratio,
                                  -b["advantages"] * ratio.clamp(1-self.clip_param, 1+self.clip_param)).mean()
        value_error = (value - b["returns"]).square()
        if self.use_clipped_value_loss:
            clipped = b["values"] + (value - b["values"]).clamp(-self.clip_param, self.clip_param)
            value_error = torch.maximum(value_error, (clipped - b["returns"]).square())
        value_loss = value_error.mean()
        smooth = mu.sum() * 0.
        pairs = ~b["dones"] & b["next_student"]
        if pairs.any():
            weights = (torch.rand_like(value[pairs, None]) - .5) * 2.
            obs = b["observations"][pairs]
            critic = b["critic_observations"][pairs]
            mixed_obs = obs + weights * (b["next_observations"][pairs] - obs)
            mixed_critic = critic + weights * (b["next_critic_observations"][pairs] - critic)
            coef = self.smoothness_upper_bound * self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
            smooth = coef * ((mu[pairs] - ac.act_inference(mixed_obs)).square().sum(-1).mean() +
                             self.value_smoothness_coef * (value[pairs] - ac.evaluate(mixed_critic).squeeze(-1)).square().mean())
        loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy + smooth
        return loss, dict(surrogate=surrogate, value=value_loss, entropy=entropy, kl=kl, smoothness=smooth)

    def update(self, storage):
        count = int(storage.data["student_valid"][:storage.step].sum())
        policy_samples = storage.amp_samples()
        metrics = dict(student_valid_count=count, amp_valid_count=len(policy_samples), updated=0)
        self.optimizer.zero_grad(set_to_none=True)
        if count == 0:
            storage.clear()
            return metrics
        totals, batches = {}, 0
        for batch in storage.batches(self.num_mini_batches, self.num_learning_epochs):
            loss, losses = self.ppo_loss(batch)
            kl = losses["kl"].item()
            if self.schedule == "adaptive" and self.desired_kl is not None:
                if kl > 2 * self.desired_kl:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0 < kl < self.desired_kl / 2:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
            # Independent AMP selection; PPO minibatch membership is irrelevant.
            if self.amp is not None and len(policy_samples):
                ids = torch.randint(len(policy_samples), (min(self.amp_batch_size, len(policy_samples)),), device=policy_samples.device)
                policy = policy_samples[ids]
                expert = self.motion_buffer.get_expert_obs(len(ids)).to(policy.device)
                if self.amp_normalizer is not None:
                    # Update with RAW observations, never normalized tensors.
                    self.amp_normalizer.update(policy)
                    self.amp_normalizer.update(expert)
                    policy = self.amp_normalizer.normalize(policy)
                    expert = self.amp_normalizer.normalize(expert)
                amp_loss, expert_loss, policy_loss = self.amp.compute_loss(policy, expert)
                loss = loss + amp_loss
                with torch.no_grad():
                    policy_output, expert_output = self.amp(policy), self.amp(expert)
                losses.update(amp_policy_loss=policy_loss, amp_expert_loss=expert_loss,
                              amp_gradient_penalty=amp_loss-expert_loss-policy_loss,
                              amp_policy_output=policy_output.mean(), amp_expert_output=expert_output.mean(),
                              amp_policy_output_std=policy_output.std(unbiased=False),
                              amp_expert_output_std=expert_output.std(unbiased=False))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite mixed handoff PPO/AMP loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, module in (("actor", self.actor_critic.actor), ("critic", self.actor_critic.critic), ("amp", self.amp)):
                if module is None:
                    continue
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Non-finite " + name + " gradients")
                losses[name + "_gradient_norm"] = torch.sqrt(sum((g.square().sum() for g in gradients), loss.new_zeros(())))
            parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
            losses["gradient_norm"] = nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            if not torch.isfinite(losses["gradient_norm"]):
                raise FloatingPointError("Non-finite combined Student/AMP gradient norm")
            self.optimizer.step()
            # ActorCritic uses a directly optimized std (not log_std).
            with torch.no_grad():
                self.actor_critic.std.clamp_(min=1e-4)
            self.optimizer_steps += 1
            batches += 1
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.) + v.detach().item()
        self.optimizer_updates += 1
        metrics.update({k: v / batches for k, v in totals.items()})
        metrics.update(updated=1, learning_rate=self.learning_rate,
                       advantage_mean=storage.advantage_mean, advantage_std=storage.advantage_std)
        storage.clear()
        return metrics
