"""
weathprov

This module implements the method for analysing major elemental geochemistry described in 
Lipp et al. (2020) ``Major Element Composition of Sediments in Termsof Weathering and Provenance: 
Implicationsfor Crustal Recycling''.

Contents:
- coeff_to_comp: Converts between omega and psi coefficients into major element compositions.
- WeathProv: Fits a compositional dataset to weath-prov model 

Citation: Lipp, A. G., Shorttle, O., Syvret, F.,& Roberts, G. G. (2020). Major element composition
of sediments interms of weathering and provenance: implications for crustal recycling.
Geochemistry, Geophysics, Geosystems,21, e2019GC008758. https://doi.org/10.1029/2019GC00875

"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compysitional.transforms import clr_df

UCC_MAJORS_CLR = pd.Series(
    {
        "SiO2": 3.053662,
        "Al2O3": 1.589025,
        "Fe2O3T": 0.577424,
        "MgO": -0.237084,
        "Na2O": 0.039447,
        "CaO": 0.132809,
        "K2O": -0.115724,
        "TiO2": -1.59163,
        "MnO": -3.447928,
    }
)  # CLR vector for Upper Continental Crust after Rudnick and Gao (2003)
PROVENANCE_VECTOR = pd.Series(
    {
        "SiO2": -0.262876,
        "Al2O3": -0.129283,
        "Fe2O3T": 0.194968,
        "MgO": 0.554626,
        "Na2O": -0.276012,
        "CaO": 0.296453,
        "K2O": -0.612535,
        "TiO2": 0.087662,
        "MnO": 0.146997,
    }
)
# Compositional trend for protolith changes (1st PC of Crater Lake suite as per Lipp et al 2020)
WEATHERING_VECTOR = pd.Series(
    {
        "SiO2": 0.224741,
        "Al2O3": 0.386040,
        "Fe2O3T": 0.124464,
        "MgO": 0.073625,
        "Na2O": -0.546488,
        "CaO": -0.653688,
        "K2O": 0.163839,
        "TiO2": 0.134725,
        "MnO": 0.092742,
    }
)  # Compositional trend for chemical weathering (1st PC of Toorongo suite as per Lipp et al 2020)
MAJOR_OXIDES = [
    "SiO2",
    "Al2O3",
    "Fe2O3T",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "TiO2",
    "MnO",
]  # Strings for major element oxides
PRISTINE_OMEGA = -0.271  # Omega value for pristine unweathered rocks
# TODO: UPDATE THIS USING UPDATED WEATHERING VECTOR


def coeff_to_comp(omega: float, psi: float):
    """Takes omega and psi coefficients and calculates corresponding major element
    composition
    """
    p = PROVENANCE_VECTOR / np.linalg.norm(PROVENANCE_VECTOR)
    w = WEATHERING_VECTOR / np.linalg.norm(WEATHERING_VECTOR)
    ucc_clr = UCC_MAJORS_CLR

    clr_out = ucc_clr + p * psi + w * omega
    comp_out = np.exp(clr_out) / np.sum(np.exp(clr_out))
    return comp_out


class WeathProv:
    """Decomposes major element composition of a solid geological
    material into contributions from weathering intensity and protolith as per
    Lipp et al. 2020, G-cubed. Can sensibly be applied to major element compositions
    of sediments, igneous, metamorphic rocks etc.

    Attributes:
        major_elements : Inputted raw compositional data
        clr : clr transform of major_elements
        coefficients : Fitted omega and psi coefficients
        fitted : Modelled compositions considering only weathering and provenance
        residuals : Residuals to model fit
        r_squared : R-squared of model fit
        protoliths : Calculates protoliths for modelled compositions

    Methods:
        fit(): Fits the omega-psi model to provided compositional data, setting most attributes
        plot(): Visualises the calculated coefficients.

    Example usage:
        >>> import pandas as pd
        >>> # Load Toorongo soil-profile data from Nesbitt & Young (1989)
        >>> toorongo = pd.DataFrame({
                "SiO2": [64.26, 64.34, 64.22, 64.27, 64.33, 64.01, 63.43, 61.08, 57.98, 59.56, 60.03, 57.58, 59.3, 49.56, 53.13],
                "Al2O3": [15.67, 15.59, 15.65, 15.68, 15.45, 15.8, 15.51, 17.71, 19.76, 18.24, 18.36, 20.03, 19.21, 28.2, 29.99],
                "Fe2O3T": [5.590062708, 5.567835818, 5.456701371, 5.645629932, 5.490041705, 5.645629932, 5.423361037, 5.167751808, 4.956596357, 5.212205587, 4.956596357, 4.967709802, 4.745440907, 2.811701521, 2.256029284],
                "MgO": [2.63, 2.64, 2.52, 2.63, 2.63, 2.62, 2.59, 2.31, 2.22, 2.21, 2.1, 2.07, 1.88, 1.01, 0.86],
                "Na2O": [3.44, 3.41, 3.33, 3.26, 3.28, 3.32, 3.11, 2.09, 1.85, 1.25, 0.47, 0.43, 0.25, 0.09, 0.07],
                "CaO": [4.31, 4.27, 3.99, 3.94, 3.7, 3.72, 3.3, 2.15, 1.33, 1.06, 0.6, 0.54, 0.34, 0.07, 0.03],
                "K2O": [2.58, 2.63, 2.53, 2.52, 2.46, 2.63, 2.65, 2.1, 2.23, 2.4, 2.48, 2.44, 2.28, 1.53, 1.3]})
        >>> weath_prov = WeathProv(comp_dataframe)
        >>> weath_prov.fit()
        >>> weath_prov.plot()

    Args:
        comp_dataframe : DataFrame of compositional data that contains major elements.

    - Note that comp_dataframe must contain the following columns: "SiO2", "Al2O3", "Fe2O3T", "MgO", "CaO", "Na2O", "K2O"
    """

    pristine: pd.Series = PRISTINE_OMEGA
    p: pd.Series = PROVENANCE_VECTOR / np.linalg.norm(PROVENANCE_VECTOR)
    w: pd.Series = WEATHERING_VECTOR / np.linalg.norm(WEATHERING_VECTOR)
    ucc_clr: pd.Series = UCC_MAJORS_CLR

    def __init__(self, comp_dataframe: pd.DataFrame) -> None:
        missing_columns = set(MAJOR_OXIDES) - set(comp_dataframe.columns)
        if missing_columns:
            missing_columns_str = ", ".join(missing_columns)
            raise Exception(
                f"The following columns are missing in the dataframe: {missing_columns_str}."
            )

        if comp_dataframe.isna().any().any():
            raise Exception("Warning: dataframe contains NA values")
        if comp_dataframe.isnull().any().any():
            raise Exception("Warning: dataframe contains Null values")
        if (comp_dataframe == 0).any().any():
            raise Exception("Warning: dataframe contains zero values")

        self.major_elements: pd.DataFrame = comp_dataframe[MAJOR_OXIDES]
        self.clr: pd.DataFrame = clr_df(self.major_elements)
        self.coefficients: pd.DataFrame = None
        self.fitted: pd.DataFrame = None
        self.residuals: pd.DataFrame = None
        self.r_squared: float = None
        self.protoliths: pd.DataFrame = None

    def _clr_to_coeffs(self, x: pd.Series) -> Tuple[float, float]:
        """Converts a clr major element vector and calculates
        the corresponding omega psi coefficients. Variable naming
        follows that from Lipp et al. 2020"""

        x_UCC = x - self.ucc_clr
        a_hat = self.w / np.linalg.norm(self.w)
        b = self.p - (a_hat.dot(self.p) * a_hat)
        b_hat = b / np.linalg.norm(b)

        alpha = x_UCC.dot(a_hat)
        beta = x_UCC.dot(b_hat)

        omega = alpha - (beta / np.linalg.norm(b)) * self.w.dot(self.p)
        psi = beta / np.linalg.norm(b)
        return omega, psi

    def _coeffs_to_clr(self, omega: float, psi: float) -> pd.Series:
        """Converts omega psi coefficents into clr vector"""
        return self.ucc_clr + self.p * psi + self.w * omega

    def _calculate_r_squared(self):
        """Calculates R-squared of model fit to data as ratio
        of fitted sum squares to total observed sum squares"""
        obs_rltv_mean = self.clr.sub(self.clr.mean(axis=0))
        fit_rltv_mean = self.fitted.sub(self.clr.mean(axis=0))
        obs_sum_square = (obs_rltv_mean**2).sum().sum()
        fit_rltv_mean = (fit_rltv_mean**2).sum().sum()
        return fit_rltv_mean / obs_sum_square

    def _get_protoliths(self) -> pd.DataFrame:
        """Returns the modelled protoliths of dataset"""
        return self.coefficients["psi"].apply(lambda psi: coeff_to_comp(self.pristine, psi))

    def fit(self) -> None:
        """Fits data to weathering-provenance model, calculating omega
        and psi coefficients for each row. Sets the following attributes of
        WeathProv object: coefficients, fitted, residuals, r-squared."""

        result = self.clr.apply(self._clr_to_coeffs, axis=1)
        self.coefficients = pd.DataFrame(
            {"omega": result.apply(lambda x: x[0]), "psi": result.apply(lambda x: x[1])}
        )
        self.fitted = self.coefficients.apply(
            lambda row: self._coeffs_to_clr(row["omega"], row["psi"]), axis=1
        )
        self.residuals = self.clr - self.fitted
        self.r_squared = self._calculate_r_squared()
        self.protoliths = self._get_protoliths()

    def plot(self) -> None:
        """Makes a labelled omega-psi plot for fitted dataset"""
        coeffs = self.coefficients
        plt.vlines(x=self.pristine, ymax=3, ymin=-3, colors="grey", linestyles="dashed")
        plt.text(x=self.pristine - 0.3, y=2.75, s="$\omega_0$", c="grey", rotation=90)
        plt.vlines(x=0, ymax=3, ymin=-3, colors="k")
        plt.text(x=0.05, y=2.6, s="UCC", c="k", rotation=90)
        plt.hlines(y=0, xmin=-1, xmax=7, colors="k")
        plt.text(x=6.5, y=0.05, s="UCC", c="k")
        plt.scatter(x=coeffs["omega"], y=coeffs["psi"])
        plt.xlabel("$\omega$, weathering intensity")
        plt.ylabel("$\psi$, protolith")
        plt.gca().set_aspect("equal")
