import torch

rollout_len = 20

rewards = torch.zeros(rollout_len)
rewards[-1] = 1.
terminals = torch.zeros(rollout_len)
terminals[-1] = 1.
terminal_masks = 1. - terminals
bootstrap_value = 0.

target = bootstrap_value
nsteps = []
for i in reversed(range(rollout_len)):
    target = rewards[i] + 0.99 * target * terminal_masks[i]
    nsteps.append(target)

print(nsteps)