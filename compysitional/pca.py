import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from compysitional.transforms import clr_df


class CompositionalPCA:
    """Class to perform PCA on a compositional dataset

    dataframe : Dataset of compositions (raw values e.g., %, ppm) with rows as compositions
    and variables as columns.
    categories : Series of categories which classify the compositions (e.g., lithology). Used to
    colour PCA plots
    values : Continuous values which might be used to classify the compositions (e.g., age)"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        categories: pd.Series = None,
        values: pd.Series = None,
    ) -> None:
        if dataframe.isna().any().any():
            raise Exception("Warning: dataframe contains NA values")
        if dataframe.isnull().any().any():
            raise Exception("Warning: dataframe contains Null values")
        if (dataframe == 0).any().any():
            raise Exception("Warning: dataframe contains zero values")
        if (categories is not None) and (values is not None):
            raise Exception("Warning cannot supply both values and categories")

        self.comp_df = dataframe
        self.categories = categories
        self.values = values
        self.clr_df = clr_df(dataframe)
        self.pca = None
        self.loadings = None
        self.scores = None

    def fit_pca(self):
        """Fit the PCA transform"""
        self.pca = PCA()
        self.pca.fit(self.clr_df)
        self.scores = pd.DataFrame(self.pca.transform(self.clr_df))
        self.loadings = pd.DataFrame(self.pca.components_)
        self.loadings.columns = self.comp_df.columns

    def plot_variance_explained(self):
        """Plot variance explored scree plot"""
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance_ratio.cumsum()

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(explained_variance_ratio) + 1),
            cumulative_variance,
            marker="o",
            linestyle="--",
        )
        plt.ylim(0, 1)
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance vs. Number of Principal Components")
        plt.grid(True)
        plt.show()

    def plot_loadings(self, x_component: int = 0, y_component: int = 1) -> None:
        """Plots loadings of dataset.
        x/y_component : The index of the components to be plotted on x and y axes"""
        for i in range(self.loadings.shape[0]):
            x, y = (
                self.loadings.iloc[x_component, i],
                self.loadings.iloc[y_component, i],
            )
            elem = self.loadings.columns[i]
            plt.arrow(
                x=0,
                y=0,
                dx=x,
                dy=y,
                head_width=0.02,
                head_length=0.05,
                fc="white",
                ec="k",
                # alpha=0.5,
                label="_nolegend_",
            )
            plt.text(x * 1.1, y * 1.1, s=elem, c="k")
        plt.xlabel(f"PC{x_component + 1}")
        plt.ylabel(f"PC{y_component + 1}")
        plt.title("PCA Loadings Plot")
        plt.gca().set_aspect("equal")

    def plot_scores(
        self,
        x_component: int = 0,
        y_component: int = 1,
        plot_means: bool = False,
        size: float = 1,
    ) -> None:
        """Plots a scores of a PCA dataset. Depending if categories and
        x/y_component : The index of the components to be plotted on x and y axes
        plot_means : Indicates if means of categories to be plotted or all data points
        size : size of markers"""

        if plot_means and (self.categories is None):
            raise Exception("`plot_means` set to True but categories not provided")
        plt.figure(figsize=(8, 6))

        if self.values is not None:
            # Colour scatter by provided values
            plt.scatter(
                self.scores[x_component],
                self.scores[y_component],
                c=self.values,
                cmap="viridis",
                alpha=0.5,
                s=size,
            )
            cb = plt.colorbar()
            cb.set_label(self.values.name)
        elif self.categories is not None:
            # Colour scatter by provided categories
            unique_categories = self.categories.unique()
            for category in unique_categories:
                if pd.isnull(category):
                    if plot_means:
                        # Ignore nan values if means being plotted
                        continue
                    category_indices = self.categories[self.categories.isnull()].index
                    plt.scatter(
                        self.scores.loc[category_indices, x_component],
                        self.scores.loc[category_indices, y_component],
                        label="NaN",
                        c="gray",
                        alpha=0.5,
                        s=size,
                    )
                else:
                    category_indices = self.categories[self.categories == category].index

                    # Calculate mean and standard deviations
                    mean_x = self.scores.loc[category_indices, x_component].mean()
                    mean_y = self.scores.loc[category_indices, y_component].mean()
                    std_x = self.scores.loc[category_indices, x_component].std()
                    std_y = self.scores.loc[category_indices, y_component].std()

                    if plot_means:
                        # Plot mean as a large square
                        plt.scatter(
                            mean_x,
                            mean_y,
                            marker="s",
                            s=200,
                            label=str(category),
                        )

                        # Plot standard deviations as whiskers
                        plt.plot(
                            [mean_x - std_x, mean_x + std_x],
                            [mean_y, mean_y],
                            color="grey",
                            linestyle="-",
                            alpha=0.7,
                        )
                        plt.plot(
                            [mean_x, mean_x],
                            [mean_y - std_y, mean_y + std_y],
                            color="grey",
                            linestyle="-",
                            alpha=0.7,
                        )
                    else:
                        plt.scatter(
                            self.scores.loc[category_indices, x_component],
                            self.scores.loc[category_indices, y_component],
                            label=str(category),
                            alpha=0.5,
                            s=size,
                        )

            plt.legend()
        else:
            plt.scatter(self.scores[x_component], self.scores[y_component], alpha=0.5, s=0.5)

        plt.xlabel(f"PC{x_component + 1}")
        plt.ylabel(f"PC{y_component + 1}")
        plt.title("PCA Scores Scatter Plot")
        plt.gca().set_aspect("equal")
