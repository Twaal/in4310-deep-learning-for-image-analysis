import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Adjust matplotlib’s style parameters to mimic a seaborn look
plt.rcParams.update({
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '0.9',
    'grid.linestyle': '-',
    'grid.linewidth': 1,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def plot_loss(save_path, losses, epoch_start_steps):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    steps = range(len(losses))
    ax1.plot(steps, losses, label='Training Loss (per step)', color='tab:blue')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))  # MultipleLocator(50))
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs. Steps (with Epochs)')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a secondary x-axis for epochs.
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Important for aligning ax2 with ax1
    # Lower the z-order of ax2 so that ax1 is drawn on top.
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)  # Set the patch of ax1 to invisible so that ax2 lines are visible
    # Set the ticks and label and stuff
    ax2.set_xticks(epoch_start_steps)
    ax2.set_xticklabels([str(epoch) for epoch in range(len(epoch_start_steps))])
    ax2.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metrics(save_path, metrics: dict, num_epochs: int):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim([0, 1])
    epochs = range(num_epochs)
    for metric_name, scores in metrics.items():
        ax.plot(epochs, scores, marker='o', label=metric_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Score')
    ax.set_title('Validation Metrics per Epoch')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    import random

    _losses = sorted([random.random() for i in range(500)], reverse=True)
    _epoch_start_steps = list(range(0, 501, 100))
    plot_loss('try_loss.png', _losses, _epoch_start_steps)

    _metrics = {
        'BLEU@4': sorted([random.random() for i in range(10)]),
        'CIDEr': sorted([random.random() for i in range(10)]),
        'ROGUE-L': sorted([random.random() for i in range(10)]),
    }
    plot_metrics('try_metrics.png', _metrics, 10)
