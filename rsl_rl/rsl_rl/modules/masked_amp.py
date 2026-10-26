"""The shared, frame-wise PhysHSI AMP projection and history ownership."""
import torch


KEEP_INDICES = tuple(range(48)) + tuple(range(54, 60))


def project_amp(observations):
    if observations.shape[-1] != 600:
        raise ValueError("Expected ten chronological 60D PhysHSI AMP frames")
    frames = observations.reshape(*observations.shape[:-1], 10, 60)
    return frames[..., list(KEEP_INDICES)].flatten(-2)


class StudentAMPHistory:
    def __init__(self, num_envs, device):
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def step(self, student, dones):
        # The returned history is post-physics, BEFORE reset. A terminal tenth
        # Student frame is valid; the next episode starts from zero.
        self.count = torch.where(student, self.count + 1, 0)
        valid = student & (self.count >= 10)
        self.count[dones] = 0
        return valid
