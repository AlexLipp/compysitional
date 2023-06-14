#!/usr/bin/env python3
"""Demonstrates use of WeathProv class by calculating 
omega and psi coefficients for Toorongo soil profile
"""

import compysitional as cp
import pandas as pd
import matplotlib.pyplot as plt

toorongo_df = pd.read_csv("eg_data/toorongo_soil.csv", index_col="Sample").iloc[:, :9]
toorongo = cp.composition.compositions_from_dataframe(toorongo_df)
toorongo_wp = cp.weathprov.WeathProv(toorongo)
toorongo_wp.fit()
print(f"R-squared for model: {toorongo_wp.r_squared}")
toorongo_wp.plot()
plt.title("Toorongo Soil Profile")
plt.show()
