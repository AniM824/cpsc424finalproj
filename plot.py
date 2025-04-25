import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_pca_on_axis(ax, df, title):
    # Identify PC columns and labels
    pc_cols = [c for c in df.columns if c.startswith('PC')]
    n_dims  = len(pc_cols)
    labels  = df['label'].astype(int)
    coords  = [df[c] for c in pc_cols]
    
    # 2D vs 3D scatter
    if n_dims == 2:
        sc = ax.scatter(coords[0], coords[1],
                        c=labels, cmap='tab10', s=1)
        ax.set_xlabel(pc_cols[0])
        ax.set_ylabel(pc_cols[1])
    elif n_dims >= 3:
        sc = ax.scatter(coords[0], coords[1], coords[2],
                        c=labels, cmap='tab10', s=1)
        ax.set_xlabel(pc_cols[0])
        ax.set_ylabel(pc_cols[1])
        ax.set_zlabel(pc_cols[2])
    else:
        raise ValueError(f"Only 2D or 3D PCA supported, got {n_dims}D")
    
    ax.set_title(f"{title} ({n_dims}D)")
    return sc

def plot_comparison(files, titles, output_filename):
    """Plot PCA results comparing different implementations."""
    # create figure
    fig = plt.figure(figsize=(16, 12))  # Adjusted height for 2x2 layout
    scats = []

    # Calculate grid dimensions
    n_plots = len(files)
    n_cols = min(2, n_plots)  # Maximum 2 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    for i, (fname, title) in enumerate(zip(files, titles)):
        # load DataFrame
        df = pd.read_csv(fname)
        
        # choose projection mode
        pc_cols = [c for c in df.columns if c.startswith('PC')]
        proj_kw = {'projection': '3d'} if len(pc_cols) >= 3 else {}
        
        # Calculate row and column for 2x2 grid
        row = i // n_cols
        col = i % n_cols
        
        ax = fig.add_subplot(n_rows, n_cols, i + 1, **proj_kw)
        sc = plot_pca_on_axis(ax, df, title)
        scats.append(sc)

    # shared colorbar for all subplots
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        scats[0],
        cax=cax,
        ticks=range(10)
    )
    cbar.set_label('Digit label')

    plt.tight_layout(rect=[0, 0, 0.90, 1.0])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

def run_implementation_comparison():
    """Run the comparison between C++ and Python implementations."""
    files = ['pca_cpp.csv', 'pca_python.csv']
    titles = ['C++ PCA', 'Python PCA']
    plot_comparison(files, titles, 'pca_comparison.png')

def run_iterations_comparison():
    """Run the comparison between different iteration counts."""
    # Files from iterations_test with different iteration counts
    iter_files = [f'pca_cpp_{iters}_iters.csv' for iters in [2, 5, 10, 50]]
    titles = [f'{iters} Iterations' for iters in [2, 5, 10, 50]]
    plot_comparison(iter_files, titles, 'pca_iterations_comparison.png')

if __name__ == "__main__":
    # Run both comparisons
    run_implementation_comparison()
    run_iterations_comparison()
