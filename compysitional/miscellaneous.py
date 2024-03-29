"""
This module provides miscellaneous functions and classes that are used in geochemical 
compositional analysis

Data:
OXIDE_ELEMENT_CONVERSION_FACTORS: A dataframe containing conversion factors for converting
elemental concentrations to oxide concentrations and vice versa.

Functions:
- convert_element_to_oxide: Converts an array of elemental concentrations to oxide concentrations.
"""

import numpy as np
import pandas as pd

OXIDE_ELEMENT_CONVERSION_FACTORS = pd.DataFrame(
    {
        "Element": [
            "Ag",
            "Al",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "C",
            "Ca",
            "Cd",
            "Ce",
            "Ce",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "Dy",
            "Er",
            "Eu",
            "Fe",
            "Fe",
            "Ga",
            "Gd",
            "Ge",
            "Hf",
            "Hg",
            "Ho",
            "In",
            "Ir",
            "K",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Nd",
            "Ni",
            "Os",
            "P",
            "Pb",
            "Pb",
            "Pd",
            "Pr",
            "Pr",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sm",
            "Sn",
            "Sr",
            "Ta",
            "Tb",
            "Tb",
            "Te",
            "Th",
            "Ti",
            "Tl",
            "Tm",
            "U",
            "U",
            "U",
            "V",
            "W",
            "Y",
            "Yb",
            "Zn",
            "Zr",
        ],
        "Oxide": [
            "Ag2O",
            "Al2O3",
            "As2O3",
            "Au2O",
            "B2O3",
            "BaO",
            "BeO",
            "Bi2O5",
            "CO2",
            "CaO",
            "CdO",
            "Ce2O3",
            "CeO2",
            "CoO",
            "Cr2O3",
            "Cs2O",
            "CuO",
            "Dy2O3",
            "Er2O3",
            "Eu2O3",
            "FeO",
            "Fe2O3",
            "Ga2O3",
            "Gd2O3",
            "GeO2",
            "HfO2",
            "HgO",
            "Ho2O3",
            "In2O3",
            "IrO",
            "K2O",
            "La2O3",
            "Li2O",
            "Lu2O3",
            "MgO",
            "MnO",
            "MnO2",
            "MoO3",
            "N2O5",
            "Na2O",
            "Nb2O5",
            "Nd2O3",
            "NiO",
            "OsO",
            "P2O5",
            "PbO",
            "PbO2",
            "PdO",
            "Pr2O3",
            "Pr6O11",
            "PtO",
            "Rb2O",
            "ReO",
            "RhO",
            "RuO",
            "SO3",
            "Sb2O5",
            "Sc2O3",
            "SeO3",
            "SiO2",
            "Sm2O3",
            "SnO2",
            "SrO",
            "Ta2O5",
            "Tb2O3",
            "Tb4O7",
            "TeO3",
            "ThO2",
            "TiO2",
            "Tl2O3",
            "Tm2O3",
            "UO2",
            "UO3",
            "U3O8",
            "V2O5",
            "WO3",
            "Y2O3",
            "Yb2O3",
            "ZnO",
            "ZrO2",
        ],
        "Factor": [
            1.0741,
            1.8895,
            1.3203,
            1.0406,
            3.2202,
            1.1165,
            2.7758,
            1.1914,
            3.6644,
            1.3992,
            1.1423,
            1.1713,
            1.2284,
            1.2715,
            1.4615,
            1.0602,
            1.2518,
            1.1477,
            1.1435,
            1.1579,
            1.2865,
            1.4297,
            1.3442,
            1.1526,
            1.4408,
            1.1793,
            1.0798,
            1.1455,
            1.2091,
            1.0832,
            1.2046,
            1.1728,
            2.1527,
            1.1371,
            1.6582,
            1.2912,
            1.5825,
            1.5003,
            3.8551,
            1.3480,
            1.4305,
            1.1664,
            1.2725,
            1.0841,
            2.2916,
            1.0772,
            1.1544,
            1.1504,
            1.1703,
            1.2082,
            1.0820,
            1.0936,
            1.0859,
            1.5555,
            1.1583,
            2.4972,
            1.3284,
            1.5338,
            1.6079,
            2.1392,
            1.1596,
            1.2696,
            1.1826,
            1.2211,
            1.1510,
            1.1762,
            1.3762,
            1.1379,
            1.6681,
            1.1174,
            1.1421,
            1.1344,
            1.2017,
            1.1792,
            1.7852,
            1.2610,
            1.2699,
            1.1387,
            1.2448,
            1.3508,
        ],
    }
)


def convert_element_to_oxide(
    elemental_array: pd.Series, element: str, oxide: str, backward: bool = False
) -> pd.Series:
    """
    Converts an array of elemental concentrations to oxide concentrations. If backward is True,
    the conversion is done from oxide to elemental concentrations. The conversion is done by
    multiplying the elemental concentration by a conversion factor. The conversion factor is taken
    from the OXIDE_ELEMENT_CONVERSION_FACTORS dataframe.
    """
    rows = np.logical_and(
        OXIDE_ELEMENT_CONVERSION_FACTORS["Element"] == element,
        OXIDE_ELEMENT_CONVERSION_FACTORS["Oxide"] == oxide,
    )
    if sum(rows) == 0:
        raise ValueError(
            f"Could not find conversion factor for {element} to {oxide} conversion."
        )
    elif sum(rows) > 1:
        raise ValueError(
            f"Found multiple conversion factors for {element} to {oxide} conversion."
        )
    index = np.where(rows)[0][0]
    conversion_factor = OXIDE_ELEMENT_CONVERSION_FACTORS.loc[index, "Factor"]
    if backward:
        conversion_factor = 1 / conversion_factor
    return elemental_array * conversion_factor
