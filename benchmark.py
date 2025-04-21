import idx2numpy
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

K = 2

# paths to your files
image_file = 'train-images-idx3-ubyte'
label_file = 'train-labels-idx1-ubyte'

# load data
images = idx2numpy.convert_from_file(image_file)
labels = idx2numpy.convert_from_file(label_file)    # shape (60000,)

# flatten & center
X = images.reshape(images.shape[0], -1).astype(float)
scaler = StandardScaler(with_std=False)
X_centered = scaler.fit_transform(X)

# PCA setup
pca = PCA(n_components=K, svd_solver='randomized', random_state=0)

# ——— timing starts here ———
t0 = time.perf_counter()
X_pca = pca.fit_transform(X_centered)
t1 = time.perf_counter()
# ——— timing ends here ———

print(f"PCA fit_transform took {t1-t0:.3f} seconds")

# build DataFrame and save as before
df = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(K)]
)
df['label'] = labels
df.to_csv('pca_python.csv', index=False)
