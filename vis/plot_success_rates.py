import numpy as np
import matplotlib.pyplot as plt

# Data structure
data = {
    'Source RL policy': (0.1822916667, 0.1145089286, 0.2239583333),
    'Source RL policy (source env)': (0.8958333333, 0.875, 0.9625),
    'Data collection policy': (0.421875, 0.3333333333, 0.5104166667),
    'Target RL policy': (0.475, 0.3854166667, 0.5268973214),
    'Prior only, no WD': (0.546875, 0.515625, 0.6160714286),
    'Prior only, WD': (0.734375, 0.6875, 0.8541666667),
    'Prior+cond, no WD': (0.234375, 0.1875, 0.3385416667),
    'Prior+cond, WD': (0.6015625, 0.5, 0.7008928571),
    'BC, WD': (0.3072916667, 0.2655691964, 0.3125),
    'BC, no WD': (0.4375, 0.34375, 0.5209635417),
    'Prior+cond, transformer, no WD': (0.671875, 0.6041666667, 0.7395833333),
    'Prior, object gripper, no WD': (0.1125, 0.0625, 0.1517857143),
    'Prior, object gripper, WD': (0.23125, 0.1249023438, 0.28125),
}

# Color mapping for each method
colors = {
    'Source RL policy (target env)': '#e74c3c',  # red
    'Source RL policy (source env)': '#8b0000',  # dark red
    'Target data policy': '#f39c12',  # orange
    'Target RL policy': '#e67e22',  # dark orange
    'Prior only, no WD': '#3498db',  # blue
    'Prior only, WD': '#2980b9',  # dark blue
    'Prior+cond, no WD': '#9b59b6',  # purple
    'Prior+cond, WD': '#8e44ad',  # dark purple
    'BC, WD': '#2ecc71',  # green
    'BC, no WD': '#27ae60',  # dark green
    'Prior, object gripper, no WD': '#1abc9c',  # teal
    'Prior, object gripper, WD': '#16a085',  # dark teal
}
def plot_comparison(labels, figsize=(12, 6), title='IQM Success Rate Comparison'):
    """Plot selected methods with error bars."""
    methods = labels  # Use labels directly to preserve order
    iqm = [data[m][0] for m in methods]
    lower = [data[m][1] for m in methods]
    upper = [data[m][2] for m in methods]
    method_colors = [colors[m] for m in methods]
    
    # Calculate error bar sizes
    yerr_lower = [iqm[i] - lower[i] for i in range(len(methods))]
    yerr_upper = [upper[i] - iqm[i] for i in range(len(methods))]
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(methods))
    
    ax.bar(x, iqm, color=method_colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.errorbar(x, iqm, yerr=[yerr_lower, yerr_upper], 
                fmt='none', ecolor='black', capsize=5, capthick=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, ax
    
# plot_comparison([
#     'Source RL policy (target env)',
#     'Target data policy',
#     'Target RL policy',
#     'Source RL policy (source env)',

# ])

# plot_comparison([
#     'BC, no WD',
#     'Target RL policy',
#     'Prior only, WD',
# ])

plot_comparison([
    'Prior, object gripper, no WD',
    'Prior, object gripper, WD',
])

# plot_comparison([
#     'Target data policy',
#     'BC, WD',
#     'BC, no WD', 
# ])


# plot_comparison([
#     'Prior+cond, no WD',
#     'Prior+cond, WD',
#     'Prior only, no WD',
#     'Prior only, WD',
# ])

plt.savefig('plots/success_rates.png')
