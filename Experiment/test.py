import numpy as np
import matplotlib.pyplot as plt
import CBC_lib


def load_array(filename):
    """
    Loads a numpy array from disk.

    Parameters:
    - filename: str — path to the array file (.npy or .txt)

    Returns:
    - np.ndarray — the loaded array
    """
    if filename.endswith('.npy'):
        return np.load(filename)
    elif filename.endswith('.txt'):
        return np.loadtxt(filename)
    else:
        raise ValueError("Unsupported file format. Use .npy or .txt")
    
    import matplotlib.pyplot as plt

def plot_time_and_violin(data, fs=1e4, title=None):
    """
    Plots the given data as both a time series and a violin plot.

    Parameters:
    - data: np.ndarray — 1D or 2D array
        - If 1D: plots time series and violin of that signal.
        - If 2D: treats each row or column as a separate trace (like trials or channels).
    - fs: float or None — sampling frequency (optional, for time axis).
    - title: str or None — optional plot title.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a numpy array.")

    if data.ndim == 1:
        data = data.reshape(1, -1)  # Convert to 2D row vector

    num_signals, num_samples = data.shape if data.shape[0] < data.shape[1] else data.T.shape
    time_axis = np.arange(num_samples) / fs if fs else np.arange(num_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})

    # Time series plot
    for trace in data:
        axes[0].plot(time_axis, trace, alpha=0.7)
    axes[0].set_xlabel("Time [s]" if fs else "Sample Index")
    axes[0].set_ylabel("Signal")
    axes[0].set_title("Time Series")

    # Violin plot
    axes[1].violinplot(dataset=data.T, showmeans=True, showmedians=True)
    axes[1].set_title("Violin Plot")
    axes[1].set_xticks([])

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

    
dummy = CBC_lib.Backbone()

load = load_array('PyCBC\Load_0p1.txt')
print("Load amp.:", dummy.get_amplitude(load))
plot_time_and_violin(load, title="Shaker Response")

