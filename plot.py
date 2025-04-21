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

# files and subplot titles
files  = ['pca_cpp.csv', 'pca_python.csv']
titles = ['C++ PCA',        'Python PCA']

# create figure
fig = plt.figure(figsize=(16, 6))
scats = []

for i, (fname, title) in enumerate(zip(files, titles), start=1):
    # load DataFrame
    df = pd.read_csv(fname)
    
    # choose projection mode
    # add_subplot accepts projection='3d' or default
    pc_cols = [c for c in df.columns if c.startswith('PC')]
    proj_kw = {'projection': '3d'} if len(pc_cols) >= 3 else {}
    
    ax = fig.add_subplot(1, 2, i, **proj_kw)
    sc = plot_pca_on_axis(ax, df, title)
    scats.append(sc)

# shared colorbar for both subplots
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(
    scats[0],
    cax=cax,
    ticks=range(10)
)
cbar.set_label('Digit label')

plt.tight_layout(rect=[0, 0, 0.90, 1.0])
plt.show()
