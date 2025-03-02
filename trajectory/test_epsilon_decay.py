import matplotlib.pyplot as plt

total_steps = 50000
epsilon = 0.3
epsilon_end = 0.05
epsilon_decay = 0.99995
epsilon_decay_start = 0
epsilon_decay_step = 2
epsilon_list = []
for step in range(total_steps):
    epsilon_list.append(epsilon)
    if step >= epsilon_decay_start and step % epsilon_decay_step == 0 \
            and epsilon > epsilon_end:
        epsilon = epsilon * epsilon_decay

plt.plot([i for i in range(total_steps)], epsilon_list)
plt.plot([i for i in range(total_steps)], [0.1 for i in range(total_steps)])
plt.plot([i for i in range(total_steps)], [1 - 0.9 * (i/total_steps) for i in range(total_steps)])
plt.show()
