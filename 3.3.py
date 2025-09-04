import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define functions
def f_minimize(x, y):
    return x**2 + y**2

def f_maximize(x, y):
    return -x**2 - y**2

def optimize_and_track(func, x_init, y_init, optimizer_class, optimizer_kwargs, n_steps):
    x = torch.tensor(x_init, requires_grad=True)
    y = torch.tensor(y_init, requires_grad=True)
    optimizer = optimizer_class([x, y], **optimizer_kwargs)
    
    traj_x, traj_y = [x_init], [y_init]
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = func(x, y)
        loss.backward()
        optimizer.step()
        
        traj_x.append(x.item())
        traj_y.append(y.item())
    
    return traj_x, traj_y

def create_contour_base(func, ax, title):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    ax.contour(X, Y, Z, levels=20, alpha=0.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Part (a): Varying momentum from 0 to 0.9 (no weight decay)
print("Part (a): Varying momentum")
momentum_values = [0.0, 0.3, 0.6, 0.9]
colors = ['red', 'blue', 'green', 'orange']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, momentum in enumerate(momentum_values):
    traj_x, traj_y = optimize_and_track(
        f_minimize,
        x_init=2.0, y_init=2.0,
        optimizer_class=optim.SGD,
        optimizer_kwargs={'lr': 0.1, 'momentum': momentum},
        n_steps=50
    )
    create_contour_base(f_minimize, axes[i], f'Momentum={momentum}')
    axes[i].plot(traj_x, traj_y, '-', color=colors[i], linewidth=1, alpha=0.7)
    axes[i].scatter(traj_x[::3], traj_y[::3], c=colors[i], s=15, alpha=0.8, zorder=5)
    axes[i].plot(traj_x[0], traj_y[0], 's', color=colors[i], markersize=10,
                 markeredgecolor='black', markeredgewidth=1.5, zorder=6)
    axes[i].plot(traj_x[-1], traj_y[-1], '*', color=colors[i], markersize=12,
                 markeredgecolor='black', markeredgewidth=1, zorder=6)

plt.suptitle('SGD Optimization: Effect of Momentum\nf(x,y) = x² + y²', fontsize=16)
plt.tight_layout()
plt.savefig('sgd_momentum_row.png', dpi=300, bbox_inches='tight')
plt.show()

# Part (b): Varying momentum with weight_decay=0.1
print("Part (b): Varying momentum (with weight decay)")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, momentum in enumerate(momentum_values):
    traj_x, traj_y = optimize_and_track(
        f_minimize,
        x_init=2.0, y_init=2.0,
        optimizer_class=optim.SGD,
        optimizer_kwargs={'lr': 0.1, 'momentum': momentum, 'weight_decay': 0.1},
        n_steps=50
    )
    create_contour_base(f_minimize, axes[i], f'Weight Decay=0.1\nMomentum={momentum}')
    axes[i].plot(traj_x, traj_y, '-', color=colors[i], linewidth=1, alpha=0.7)
    axes[i].scatter(traj_x[::3], traj_y[::3], c=colors[i], s=15, alpha=0.8, zorder=5)
    axes[i].plot(traj_x[0], traj_y[0], 's', color=colors[i], markersize=10,
                 markeredgecolor='black', markeredgewidth=1.5, zorder=6)
    axes[i].plot(traj_x[-1], traj_y[-1], '*', color=colors[i], markersize=12,
                 markeredgecolor='black', markeredgewidth=1, zorder=6)

plt.suptitle('SGD Optimization: Effect of Momentum (with Weight Decay=0.1)\nf(x,y) = x² + y²', fontsize=16)
plt.tight_layout()
plt.savefig('sgd_momentum_with_weight_decay_row.png', dpi=300, bbox_inches='tight')
plt.show()

# Part (c): maximize=True comparison (just two panels in a row)
print("Part (c): Using maximize=True")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Regular minimization of -x^2 - y^2
traj_x_min, traj_y_min = optimize_and_track(
    f_maximize,
    x_init=0.5, y_init=0.5,
    optimizer_class=optim.SGD,
    optimizer_kwargs={'lr': 0.1, 'momentum': 0.0},
    n_steps=20
)
create_contour_base(f_maximize, axes[0], 'Minimize -x² - y²\n(regular minimization)')
axes[0].plot(traj_x_min, traj_y_min, '-', color="red", linewidth=1, alpha=0.7)
axes[0].scatter(traj_x_min[::2], traj_y_min[::2], c="red", s=25, alpha=0.8, zorder=5)
axes[0].plot(traj_x_min[0], traj_y_min[0], 's', color="red", markersize=10,
             markeredgecolor='black', markeredgewidth=1.5, zorder=6)
axes[0].plot(traj_x_min[-1], traj_y_min[-1], '*', color="red", markersize=12,
             markeredgecolor='black', markeredgewidth=1, zorder=6)

# Maximization of -x^2 - y^2
traj_x_max, traj_y_max = optimize_and_track(
    f_maximize,
    x_init=0.5, y_init=0.5,
    optimizer_class=optim.SGD,
    optimizer_kwargs={'lr': 0.1, 'momentum': 0.0, 'maximize': True},
    n_steps=30
)
create_contour_base(f_maximize, axes[1], 'Maximize -x² - y²\n(maximize=True)')
axes[1].plot(traj_x_max, traj_y_max, '-', color="blue", linewidth=1, alpha=0.7)
axes[1].scatter(traj_x_max[::2], traj_y_max[::2], c="blue", s=25, alpha=0.8, zorder=5)
axes[1].plot(traj_x_max[0], traj_y_max[0], 's', color="blue", markersize=10,
             markeredgecolor='black', markeredgewidth=1.5, zorder=6)
axes[1].plot(traj_x_max[-1], traj_y_max[-1], '*', color="blue", markersize=12,
             markeredgecolor='black', markeredgewidth=1, zorder=6)

plt.suptitle('SGD Optimization: Effect of maximize=True\nf(x,y) = -x² - y²', fontsize=14)
plt.tight_layout()
plt.savefig('sgd_maximize_row.png', dpi=300, bbox_inches='tight')
plt.show()