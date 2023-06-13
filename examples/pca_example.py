#!/usr/bin/env python3
"""Demonstrates use of CompositionalPCA class by extracting
compositional trends of weathering and igneous variability from 
a soil profile dataset and a suite of igneous rocks."""

import matplotlib.pyplot as plt
import pandas as pd

import compysitional as cp

toorongo = pd.read_csv("eg_data/toorongo_soil.csv").iloc[:, 1:-1]
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

craterlake = pd.read_csv("eg_data/craterlake_igneous.csv").iloc[:, 4:13]
craterlake_pca = cp.pca.CompositionalPCA(craterlake)
craterlake_pca.fit()
craterlake_pca.plot_variance_explained()
craterlake_pca.plot_loadings()
# craterlake_pca.plot_scores()
plt.title("Crater Lake Igneous Suite")
plt.show()
craterlake_pc1 = craterlake_pca.loadings.loc[0]
print("Protolith Vector")
print(craterlake_pc1)
