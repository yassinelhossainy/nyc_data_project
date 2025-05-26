import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew, chi2, kurtosis
from sqlalchemy import engine
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel.results import compare

def clean_column_names(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    return df
class StatWhiz:
    def __init__(self, data, entity, time, dep):
        self.data = data.copy()
        self.entity = entity
        self.time = time
        self.dep = dep

    def iqr_outlier_mask(self, group_cols, value_col):
        q1 = self.data.groupby(group_cols)[value_col].transform(lambda x: x.quantile(0.25))
        q3 = self.data.groupby(group_cols)[value_col].transform(lambda x: x.quantile(0.75))
        lower = q1 - 1.5 * (q3 - q1)
        upper = q3 + 1.5 * (q3 - q1)
        return (self.data[value_col] < lower) | (self.data[value_col] > upper)

    def tag_outliers(self, group_cols, fill="no"):
        exclude_cols = {self.time, self.dep}
        numeric_cols = [col for col in self.data.select_dtypes(include="number").columns if col not in exclude_cols]

        for col in numeric_cols:
            outlier_mask = self.iqr_outlier_mask(group_cols, col)
            self.data[f'{col}_outlier'] = outlier_mask

            if fill == "yes":
                replace_mask = outlier_mask | self.data[col].isna()
                group_means = self.data.groupby(group_cols)[col].transform('mean')
                self.data.loc[replace_mask, col] = group_means[replace_mask]

    def panel_regression(self, indep_vars=None, effect_type='fixed', hausman=False):
        if indep_vars is None:
            indep_vars = [col for col in self.data.columns if col not in [self.entity, self.time, self.dep]]

        df = self.data[[self.entity, self.time, self.dep] + indep_vars].dropna().copy()
        df = df.set_index([self.entity, self.time])

        if effect_type == 'fixed':
            model = PanelOLS(df[self.dep], df[indep_vars], entity_effects=True)
            results = model.fit(cov_type='clustered', cluster_entity=True)
            print("Fixed Effects Model (Cluster-Robust SE):\n", results.summary)

        elif effect_type == 'random':
            model = RandomEffects(df[self.dep], df[indep_vars])
            results = model.fit(cov_type='clustered', cluster_entity=True)
            print("Random Effects Model (Cluster-Robust SE):\n", results.summary)

        if hausman:
            from linearmodels.panel import compare
            fe_model = PanelOLS(df[self.dep], df[indep_vars], entity_effects=True).fit()
            re_model = RandomEffects(df[self.dep], df[indep_vars]).fit()
            comparison = compare({'FE': fe_model, 'RE': re_model})
            print("\nHausman Test:\n", comparison)

        return results

    def compare_se(self, indep_vars=None, effect_type='fixed'):
        if indep_vars is None:
            indep_vars = [col for col in self.data.columns if col not in [self.entity, self.time, self.dep]]

        df = self.data[[self.entity, self.time, self.dep] + indep_vars].dropna().copy()
        df = df.set_index([self.entity, self.time])

        if effect_type == 'fixed':
            model = PanelOLS(df[self.dep], df[indep_vars], entity_effects=True)
        elif effect_type == 'random':
            model = RandomEffects(df[self.dep], df[indep_vars])
        else:
            raise ValueError("effect_type must be 'fixed' or 'random'")

        results_default = model.fit()
        results_cluster = model.fit(cov_type='clustered', cluster_entity=True)

        print("=== Without Cluster-Robust SE ===")
        print(results_default.summary)
        print("\n=== With Cluster-Robust SE (clustered by entity) ===")
        print(results_cluster.summary)

        se_default = results_default.std_errors
        se_cluster = results_cluster.std_errors
        se_comparison = se_default.to_frame(name='Default SE')
        se_comparison['Cluster-Robust SE'] = se_cluster
        print("\nStandard Errors Comparison:")
        print(se_comparison)

        return results_default, results_cluster

    def test_multicollinearity(self):
        excluded_vars = {self.entity, self.time, self.dep}
        current_df = self.data.drop(columns=excluded_vars)
        numeric_cols = current_df.select_dtypes(include='number').columns.tolist()
        current_df = current_df[numeric_cols]
        X = current_df.replace([np.inf, -np.inf], np.nan).dropna()
        X = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data['variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def test_heteroskedasticity(self):
        data = self.data.copy()
        y = data[self.dep]
        X = data.drop(columns=[self.dep, self.time, self.entity])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        data['resid'] = model.resid
        data['fitted'] = model.fittedvalues

        group_vars = data.groupby(self.entity)['resid'].var()
        N = len(group_vars)
        T = data.groupby(self.entity).size().iloc[0]
        k = X.shape[1]

        sigma_sq = np.mean(group_vars)
        wald_stat = sum((group_vars - sigma_sq) ** 2) / (2 * sigma_sq ** 2 / (T - k))
        p_value = 1 - chi2.cdf(wald_stat, df=N - 1)

        print("=== Modified Wald Test for Groupwise Heteroskedasticity ===")
        print(f"Chi-squared: {wald_stat:.4f}")
        print(f"Degrees of freedom: {N - 1}")
        print(f"P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("→ Reject H0: Evidence of heteroskedasticity.")
        else:
            print("→ Fail to reject H0: No evidence of heteroskedasticity.")

        # Plot residuals vs. fitted values by entity
        unique_entities = data[self.entity].unique()
        n_cols = 5
        n_rows = int(np.ceil(len(unique_entities) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, entity in enumerate(unique_entities):
            subset = data[data[self.entity] == entity]
            axes[i].scatter(subset['fitted'], subset['resid'], alpha=0.5)
            axes[i].axhline(0, color='gray', linestyle='--')
            axes[i].set_title(f'{self.entity}: {entity}')

        for ax in axes[len(unique_entities):]:
            fig.delaxes(ax)

        fig.suptitle('Residuals vs Fitted by Entity', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
def plot_diagnostics(self, fix='n'):
    """
    Plots interactive histograms with skewness and kurtosis for all independent variables in self.data,
    excluding entity, time, and dependent variables, using Plotly.

    If fix='y', applies recommended transformations based on skewness, modifies self.data,
    and plots the transformed distributions.

    Args:
        fix (str): 'y' to apply and plot transformations, else 'n'.
    """
    exclude_vars = {self.entity, self.time, self.dep}
    indep_vars = [col for col in self.data.columns if col not in exclude_vars]

    n_cols = 3
    n_rows = math.ceil(len(indep_vars) / n_cols)

    fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=indep_vars)

    # Define transformation decision based on skewness
    def decide_transformation(skewness):
        if skewness > 2:
            return 'reciprocal'
        elif 1 < skewness <= 2:
            return 'sqrt'
        elif 0.5 < skewness <= 1:
            return 'log'
        elif -0.5 <= skewness <= 0.5:
            return 'none'
        elif -1 <= skewness < -0.5:
            return 'log (neg)'
        elif -2 <= skewness < -1:
            return 'sqrt (neg)'
        else:  # skewness < -2
            return 'reciprocal (neg)'

    # Safe transformations with shift to avoid log(0) or division by zero
    def safe_log(x):
        shift = abs(min(x.min(), 0)) + 1e-6
        return np.log(x + shift)

    def safe_sqrt(x):
        shift = abs(min(x.min(), 0)) + 1e-6
        return np.sqrt(x + shift)

    def safe_reciprocal(x):
        shift = abs(min(x.min(), 0)) + 1e-6
        return 1 / (x + shift)

    for i, var in enumerate(indep_vars):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        col_data = self.data[var].dropna()
        original_skew = skew(col_data)
        original_kurt = kurtosis(col_data)
        transform = decide_transformation(original_skew)

        # Apply transformation if fix == 'y'
        if fix == 'y' and transform != 'none':
            if 'log' in transform:
                self.data[var] = safe_log(self.data[var])
            elif 'sqrt' in transform:
                self.data[var] = safe_sqrt(self.data[var])
            elif 'reciprocal' in transform:
                self.data[var] = safe_reciprocal(self.data[var])

            col_data = self.data[var].dropna()
            new_skew = skew(col_data)
            new_kurt = kurtosis(col_data)
        else:
            new_skew, new_kurt = original_skew, original_kurt

        # Plot histogram
        fig.add_trace(
            go.Histogram(x=col_data, nbinsx=30, marker_color='skyblue', name=var, showlegend=False),
            row=row, col=col
        )

        # Annotation text with old and new metrics
        annotation_text = (f"Transform: {transform}<br>"
                           f"Skew (old): {original_skew:.2f}<br>"
                           f"Kurt (old): {original_kurt:.2f}<br>"
                           f"Skew (new): {new_skew:.2f}<br>"
                           f"Kurt (new): {new_kurt:.2f}")

        fig.add_annotation(
            x=0.95, y=0.95,
            xref=f"x{'' if i==0 else i+1} domain",
            yref=f"y{'' if i==0 else i+1} domain",
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=3
        )

    total_plots = n_rows * n_cols
    for empty_i in range(len(indep_vars), total_plots):
        fig.layout.annotations[empty_i].visible = False

    fig.update_layout(
        height=300 * n_rows,
        width=900,
        title_text="Distributions of Independent Variables (Transformed if fix='y') with Skewness & Kurtosis",
        showlegend=False,
        bargap=0.1
    )
    fig.show()