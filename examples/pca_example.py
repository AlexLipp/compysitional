#!/usr/bin/env python3
"""Demonstrates use of CompositionalPCA class by extracting
compositional trend of weathering from toorongo soil profile
"""

import compysitional as cp
import matplotlib.pyplot as plt
import pandas as pd

toorongo_df = pd.read_csv("eg_data/toorongo_soil.csv", index_col="Sample").iloc[:, :9]
toorongo = cp.composition.compositions_from_dataframe(toorongo_df)
toorongo_pca = cp.pca.CompositionalPCA(toorongo)
toorongo_pca.fit()
toorongo_pca.plot_variance_explained()
toorongo_pca.plot_loadings()
# toorongo_pca.plot_scores()
plt.title("Toorongo Soil Profile")
plt.show()
toorongo_pc1 = toorongo_pca.loadings.loc[0]
print("Weathering Vector")
print(toorongo_pc1)
