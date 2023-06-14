"""
Weathering-Provenance Module

This module provides functionality to fit a weathering-provenance model to a compositional dataset.
It includes functions to calculate coefficients, transform compositions, and plot the results. It
implements the method for analysing major elemental geochemistry described in Lipp et al. (2020) 
``Major Element Composition of Sediments in Terms of Weathering and Provenance: Implications for 
Crustal Recycling''.

Contents:
- UCC_MAJORS: Composition of Upper Continental Crust after Rudnick and Gao (2003)
- PROVENANCE_VECTOR: Compositional trend for protolith changes
- WEATHERING_VECTOR: Compositional trend for chemical weathering
- PRISTINE_OMEGA: Omega value for pristine unweathered rocks

Functions:
- coeffs_to_composition(omega: float, psi: float) -> coda.Composition:
    Calculates the major element composition based on omega and psi coefficients.
- composition_to_coeffs(composition: coda.Composition) -> Tuple[float, float]:
    Calculates the omega and psi coefficients based on a composition.

Classes:
- WeathProv:
    Weathering-Provenance Model class that fits the weathering-provenance model to a CompositionalDataset.
    Provides methods to fit data and plot the results.

Citation: Lipp, A. G., Shorttle, O., Syvret, F.,& Roberts, G. G. (2020). Major element composition
of sediments interms of weathering and provenance: implications for crustal recycling.
Geochemistry, Geophysics, Geosystems,21, e2019GC008758. https://doi.org/10.1029/2019GC00875

"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compysitional.composition as coda

UCC_MAJORS = coda.Composition(
    {
        "SiO2": 0.6628855493064029,
        "Al2O3": 0.15323387511522135,
        "Fe2O3T": 0.055721404211408435,
        "MgO": 0.024676622792010192,
        "Na2O": 0.03253730736175315,
        "CaO": 0.03572137859697889,
        "K2O": 0.02786067927553831,
        "TiO2": 0.006368158573209143,
        "MnO": 0.000995024767477488,
    }
)  # Composition of Upper Continental Crust after Rudnick and Gao (2003)
PROVENANCE_VECTOR = coda.Composition(
    {
        "SiO2": 0.08093063126589121,
        "Al2O3": 0.09249785282728326,
        "Fe2O3T": 0.12792402691074226,
        "MgO": 0.18329459679554758,
        "Na2O": 0.07987447811537915,
        "CaO": 0.14158802158689665,
        "K2O": 0.057050295623748394,
        "TiO2": 0.11490785076224003,
        "MnO": 0.12193224611227155,
    }
)
# Compositional trend for protolith changes (1st PC of Crater Lake suite as per Lipp et al 2020)
WEATHERING_VECTOR = coda.Composition(
    {
        "SiO2": 0.13247798272176506,
        "Al2O3": 0.1556664375277568,
        "Fe2O3T": 0.11983783365568589,
        "MgO": 0.11389767209128661,
        "Na2O": 0.06126368806948359,
        "CaO": 0.0550359870274761,
        "K2O": 0.12465057848330773,
        "TiO2": 0.12107382030876958,
        "MnO": 0.11609600011446866,
    }
)  # Compositional trend for chemical weathering (1st PC of Toorongo suite as per Lipp et al 2020)
# TODO: UPDATE USING BETTER CALIBRATED WEATHERING VECTORS

PRISTINE_OMEGA: float = -0.271  # Omega value for pristine unweathered rocks
# TODO: UPDATE THIS USING UPDATED WEATHERING VECTOR


def coeffs_to_composition(omega: float, psi: float) -> coda.Composition:
    """Takes omega and psi coefficients and calculates corresponding major element composition.

    Args:
        omega (float): Omega coefficient.
        psi (float): Psi coefficient.

    Returns:
        coda.Composition: Corresponding major element composition.
    """

    out = coda.add(
        coda.add(UCC_MAJORS, coda.multiply(PROVENANCE_VECTOR, psi)),
        coda.multiply(WEATHERING_VECTOR, omega),
    )
    return out


def composition_to_coeffs(composition: coda.Composition) -> Tuple[float, float]:
    """Takes a Composition and calculates the corresponding omega psi coefficients.

    Args:
        composition (coda.Composition): The composition.

    Returns:
        Tuple[float, float]: The omega and psi coefficients.
    """
    if not (UCC_MAJORS.components == composition.components):
        raise ValueError(
            f"Composition must contain exclusively the following components: {UCC_MAJORS.components}."
        )

    x_UCC = pd.Series(coda.subtract(composition, UCC_MAJORS).clr)
    w, p = pd.Series(WEATHERING_VECTOR.clr), pd.Series(PROVENANCE_VECTOR.clr)
    a_hat = w / np.linalg.norm(w)
    b = p - (a_hat.dot(p) * a_hat)
    b_hat = b / np.linalg.norm(b)

    alpha = x_UCC.dot(a_hat)
    beta = x_UCC.dot(b_hat)

    omega = alpha - (beta / np.linalg.norm(b)) * w.dot(p)
    psi = beta / np.linalg.norm(b)
    return omega, psi


class WeathProv:
    """
    Weathering-Provenance Model

    This class fits the weathering-provenance model to a CompositionalDataset.
     t provides methods to fit data to the model and plot the results.

    Attributes:
        data (coda.CompositionalDataset): The compositional dataset.
        coefficients (pd.DataFrame): Coefficients calculated for each row, including omega and psi values.
        fitted (coda.CompositionalDataset): The fitted dataset based on the model.
        residuals (coda.CompositionalDataset): The residuals obtained from the model fit.
        r_squared (float): The R-squared value indicating the quality of the model fit.
        protoliths (coda.CompositionalDataset): The protolith dataset based on the model.

    Methods:
        __init__(self, comp_data: coda.CompositionalDataset) -> None:
            Initializes the WeathProv object.

        fit(self) -> None:
            Fits data to the weathering-provenance model and sets the relevant attributes.

        plot(self) -> None:
            Makes a labeled omega-psi plot for the fitted dataset.
    """

    def __init__(self, comp_data: coda.CompositionalDataset) -> None:
        """Initializes the WeathProv object.

        Args:
            comp_data (coda.CompositionalDataset): The compositional dataset.
        """

        if not (UCC_MAJORS.components == comp_data.components):
            raise ValueError(
                f"Composition must contain exclusively the following components: {UCC_MAJORS.components}."
            )

        self.data: coda.CompositionalDataset = comp_data
        self.coefficients: pd.DataFrame = None
        self.fitted: coda.CompositionalDataset = None
        self.residuals: coda.CompositionalDataset = None
        self.r_squared: float = None
        self.protoliths: coda.CompositionalDataset = None

    def _calculate_r_squared(self):
        """Calculates R-squared of model fit to data as ratio of fitted sum squares to total observed sum squares.

        Returns:
            float: The R-squared value.
        """
        obs_rltv_mean = self.data.clr_df.sub(self.data.geometric_mean.clr)
        fit_rltv_mean = self.fitted.clr_df.sub(self.fitted.geometric_mean.clr)
        obs_sum_square = (obs_rltv_mean**2).sum().sum()
        fit_rltv_mean = (fit_rltv_mean**2).sum().sum()
        return fit_rltv_mean / obs_sum_square

    def fit(self) -> None:
        """Fits data to weathering-provenance model, calculating omega and psi coefficients for each row.
        Sets the following attributes of the WeathProv object: coefficients, fitted, residuals, r-squared.
        """
        result_coeffs = {
            name: composition_to_coeffs(comp) for name, comp in self.data.compositions.items()
        }
        self.coefficients = pd.DataFrame(result_coeffs, index=["omega", "psi"]).T
        self.fitted = coda.CompositionalDataset(
            {
                name: coeffs_to_composition(coeffs[0], coeffs[1])
                for name, coeffs in result_coeffs.items()
            }
        )
        self.residuals = coda.CompositionalDataset(
            {
                name: coda.subtract(self.data.compositions[name], self.fitted.compositions[name])
                for name in self.data.compositions
            }
        )
        self.protoliths = coda.CompositionalDataset(
            {
                name: coeffs_to_composition(PRISTINE_OMEGA, coeffs[1])
                for name, coeffs in result_coeffs.items()
            }
        )
        self.r_squared = self._calculate_r_squared()

    def plot(self) -> None:
        """Makes a labeled omega-psi plot for the fitted dataset."""
        coeffs = self.coefficients
        plt.vlines(x=PRISTINE_OMEGA, ymax=3, ymin=-3, colors="grey", linestyles="dashed")
        plt.text(x=PRISTINE_OMEGA - 0.3, y=2.75, s="$\omega_0$", c="grey", rotation=90)
        plt.vlines(x=0, ymax=3, ymin=-3, colors="k")
        plt.text(x=0.05, y=2.6, s="UCC", c="k", rotation=90)
        plt.hlines(y=0, xmin=-1, xmax=7, colors="k")
        plt.text(x=6.5, y=0.05, s="UCC", c="k")
        plt.scatter(x=coeffs["omega"], y=coeffs["psi"])
        plt.xlabel("$\omega$, weathering intensity")
        plt.ylabel("$\psi$, protolith")
        plt.gca().set_aspect("equal")
