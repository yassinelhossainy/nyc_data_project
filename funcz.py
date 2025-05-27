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
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew, chi2, kurtosis,pearsonr
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
    def correlation_matrix(self, method='pearson', figsize=(10, 8), annot=True, cmap='coolwarm'):
        """
        Plot a heatmap of the correlation matrix.
        method: 'pearson', 'spearman', or 'kendall'
        """
        numeric_df = self.data.drop(columns=[self.time]).select_dtypes(include='number')
        corr = numeric_df.corr(method=method)

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, square=True, linewidths=0.5)
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.show()
    def test_heteroskedasticity(self):

        # Prepare panel data
        data = self.data.copy()
        data = data.set_index([self.entity, self.time])
        y = pd.to_numeric(data[self.dep], errors='coerce')
        X = data.drop(columns=[self.dep])
        X = X.apply(pd.to_numeric, errors='coerce')
        X = sm.add_constant(X)

        # Drop missing values
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined[self.dep]
        X = combined.drop(columns=[self.dep])

        # Fit Fixed Effects (entity_effects=True)
        fe_model = PanelOLS(y, X, entity_effects=True)
        fe_results = fe_model.fit()

        # Get residuals
        resid = fe_results.resids
        resid = resid.dropna()
        resid.name = 'resid'
        resid_df = resid.reset_index()

        # Compute residual variance by group
        group_vars = resid_df.groupby(self.entity)['resid'].var()
        N = len(group_vars)
        T_vals = resid_df.groupby(self.entity).size()

        if T_vals.nunique() != 1:
            raise ValueError("Unbalanced panel detected. This test requires a balanced panel.")
        
        T = T_vals.iloc[0]
        k = X.shape[1]

        sigma_sq = np.mean(group_vars)
        wald_stat = sum((group_vars - sigma_sq) ** 2) / (2 * sigma_sq ** 2 / (T - k))
        p_value = 1 - chi2.cdf(wald_stat, df=N - 1)

        print("=== Modified Wald Test for Groupwise Heteroskedasticity (FE Residuals) ===")
        print(f"Chi-squared: {wald_stat:.4f}")
        print(f"Degrees of freedom: {N - 1}")
        print(f"P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("→ Reject H0: Evidence of heteroskedasticity.")
        else:
            print("→ Fail to reject H0: No evidence of heteroskedasticity.")

        # Plot residuals vs fitted (from FE model)
        fitted = fe_results.predict().dropna()
        fitted.name = 'fitted'
        plot_df = pd.concat([resid, fitted], axis=1).reset_index()

        unique_entities = plot_df[self.entity].unique()
        n_cols = 5
        n_rows = int(np.ceil(len(unique_entities) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, entity in enumerate(unique_entities):
            subset = plot_df[plot_df[self.entity] == entity]
            axes[i].scatter(subset['fitted'], subset['resid'], alpha=0.5)
            axes[i].axhline(0, color='gray', linestyle='--')
            axes[i].set_title(f'{entity}')

        for ax in axes[len(unique_entities):]:
            fig.delaxes(ax)

        fig.suptitle('Residuals vs Fitted by Entity (Fixed Effects)', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    def add_polynomial_terms(self, columns, degree=2):
        """
        Adds polynomial terms up to the specified degree for given columns in self.data DataFrame.
        Modifies self.data in place.

        Parameters:
        - columns: list of column names to add polynomial terms for
        - degree: highest degree of polynomial terms (default=2 for squares)
        """
        for col in columns:
            for deg in range(2, degree + 1):
                new_col_name = f"{col}_pow_{deg}"
                self.data[new_col_name] = self.data[col] ** deg
    def plot_diagnostics(self, fix='n', skew_bound=1):
        """
        Plots interactive histograms with skewness and kurtosis for all numeric independent variables in self.data,
        excluding entity, time, and dependent variables, using Plotly.

        If fix='y', applies log transformations for skewed data and plots the transformed distributions.
        """
        exclude_vars = {self.entity, self.time, self.dep}
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns
        indep_vars = [col for col in numeric_vars if col not in exclude_vars]

        if not indep_vars:
            print("No numeric independent variables to plot.")
            return

        n_cols = 3
        n_rows = math.ceil(len(indep_vars) / n_cols)

        fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=indep_vars)

        def decide_transformation(skewness):
            if skewness > skew_bound:
                return 'log'
            elif skewness < -1 * skew_bound:
                return 'log_neg'
            else:
                return 'none'

        def safe_log(x):
            shift = abs(min(x.min(), 0)) + 1e-6
            return np.log(x + shift)

        def safe_log_neg(x):
            # Reflect then apply log: log(max(x) - x + 1)
            reflected = x.max() - x
            shift = abs(min(reflected.min(), 0)) + 1e-6
            return np.log(reflected + shift)

        for i, var in enumerate(indep_vars):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1

            col_data = self.data[var].dropna()
            if col_data.empty:
                continue

            original_skew = skew(col_data)
            original_kurt = kurtosis(col_data)
            transform = decide_transformation(original_skew)

            # Apply transformation if fix == 'y'
            if fix == 'y' and transform != 'none':
                if transform == 'log':
                    self.data[var] = safe_log(self.data[var])
                elif transform == 'log_neg':
                    self.data[var] = safe_log_neg(self.data[var])
                col_data = self.data[var].dropna()
                new_skew = skew(col_data)
                new_kurt = kurtosis(col_data)
            else:
                new_skew = original_skew
                new_kurt = original_kurt

            fig.add_trace(
                go.Histogram(x=col_data, nbinsx=30, marker_color='skyblue', name=var, showlegend=False),
                row=row, col=col
            )

            if fix == 'y':
                annotation_text = (
                    f"Transform: {transform}<br>"
                    f"Skew (old): {original_skew:.2f}<br>"
                    f"Kurt (old): {original_kurt:.2f}<br>"
                    f"Skew (new): {new_skew:.2f}<br>"
                    f"Kurt (new): {new_kurt:.2f}"
                )
            else:
                annotation_text = f"Transform: {transform}<br>Skew: {original_skew:.2f}<br>Kurt: {original_kurt:.2f}"

            fig.add_annotation(
                x=0.95, y=0.95,
                xref="x domain" if i == 0 else f"x{i+1} domain",
                yref="y domain" if i == 0 else f"y{i+1} domain",
                text=annotation_text,
                showarrow=False,
                font=dict(size=10),
                align="right",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=3
            )

        fig.update_layout(
            height=300 * n_rows,
            width=900,
            title_text="Distributions of Independent Variables (Transformed if fix='y') with Skewness & Kurtosis",
            showlegend=False,
            bargap=0.1
        )
        fig.show()
    def zscore_standardize(self, columns=None):
        """
        Apply Z-score standardization to specified columns in self.df.

        Parameters:
            columns (list or None): List of column names to standardize.
                                    If None, all numeric columns are standardized.
        """
    

    
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
    def plot_scatter_with_dep(self, n_cols=2):
        """
        Creates scatterplots of all numeric independent variables against the dependent variable.
        Arranges subplots in a grid with specified number of columns.
        Each subplot shows Pearson correlation coefficient in its title.

        Parameters:
            n_cols (int): Number of columns in the subplot layout.
        """
        exclude_vars = {self.entity, self.time, self.dep}
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns
        indep_vars = [col for col in numeric_vars if col not in exclude_vars]

        if not indep_vars:
            print("No numeric independent variables to plot.")
            return

        n_rows = math.ceil(len(indep_vars) / n_cols)

        # Calculate correlations for titles
        titles = []
        for var in indep_vars:
            x = self.data[var]
            y = self.data[self.dep]
            try:
                r, _ = pearsonr(x, y)
                titles.append(f"{var} (r = {r:.2f})")
            except Exception:
                titles.append(f"{var} (r = N/A)")

        fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

        for i, var in enumerate(indep_vars):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1

            x = self.data[var]
            y = self.data[self.dep]

            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(color='blue', opacity=0.6),
                    name=var,
                    showlegend=False
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text=var, row=row, col=col)
            fig.update_yaxes(title_text=self.dep, row=row, col=col)

        fig.update_layout(
            height=300 * n_rows,
            width=350 * n_cols,
            title_text=f"Scatterplots of Independent Variables vs {self.dep}",
            showlegend=False,
        )
        fig.show()
    def compare_same_type_models(self, model_specs, effect_type='fixed', robust=True, het_test=True):
        df = self.data.copy()
        df = df.set_index([self.entity, self.time])

        results_dict = {}
        aic_bic_summary = []
        het_pvalues = {}

        for spec in model_specs:
            name = spec['name']
            indep_vars = spec['indep_vars']

            data = df[[self.dep] + indep_vars].dropna()
            y = data[self.dep]
            X = sm.add_constant(data[indep_vars])

            if effect_type == 'fixed':
                model = PanelOLS(y, X, entity_effects=True)
            elif effect_type == 'random':
                model = RandomEffects(y, X)
            else:
                raise ValueError("Invalid effect_type. Choose 'fixed' or 'random'.")

            cov_type = 'clustered' if robust else 'unadjusted'
            results = model.fit(cov_type=cov_type, cluster_entity=robust)
            results_dict[name] = results

            n = results.nobs
            rss = np.sum(results.resids**2)
            llf = -n / 2 * (np.log(rss / n) + 1)

            k = results.params.shape[0]
            aic = -2 * llf + 2 * k
            bic = -2 * llf + np.log(n) * k

            aic_bic_summary.append({'Model': name, 'AIC': aic, 'BIC': bic})

            if het_test:
                test = het_breuschpagan(results.resids, X)
                bp_stat, bp_pvalue, f_stat, f_pvalue = test
                het_pvalues[name] = {'Breusch-Pagan p-value': bp_pvalue}

                if bp_pvalue < 0.05:
                    print(f"\n*** Heteroskedasticity detected in model '{name}' (BP p = {bp_pvalue:.4f}) ***")
                    print("Parameter p-values:")
                    print(results.pvalues.round(4))
                else:
                    print(f"\nNo heteroskedasticity detected in model '{name}' (BP p = {bp_pvalue:.4f})")

        comparison = compare(results_dict)
        print(f"\n=== Comparing {effect_type.title()} Effects Models ===")
        print(comparison)

        print("\nAIC and BIC:")
        print(pd.DataFrame(aic_bic_summary).set_index('Model'))

        if het_test:
            print("\nBreusch-Pagan Test p-values:")
            print(pd.DataFrame(het_pvalues).T)

        return comparison
    def fit_and_plot_residuals(self, indep_vars=None, effect_type='fixed', robust=True):
        """
        Fits a panel regression model and plots residuals vs. fitted values for each entity.

        Parameters:
            indep_vars (list): List of independent variables. If None, automatically inferred.
            effect_type (str): 'fixed' or 'random'.
            robust (bool): If True, uses cluster-robust standard errors.

        Returns:
            results: Fitted model results
        """
        if indep_vars is None:
            indep_vars = [col for col in self.data.columns if col not in [self.entity, self.time, self.dep]]

        # Prepare data
        df = self.data[[self.entity, self.time, self.dep] + indep_vars].dropna().copy()
        df = df.set_index([self.entity, self.time])
        y = df[self.dep]
        X = df[indep_vars]

        # Fit model
        if effect_type == 'fixed':
            model = PanelOLS(y, X, entity_effects=True)
        elif effect_type == 'random':
            model = RandomEffects(y, X)
        else:
            raise ValueError("effect_type must be 'fixed' or 'random'")

        cov_type = 'clustered' if robust else 'unadjusted'
        results = model.fit(cov_type=cov_type, cluster_entity=robust)

        print(results.summary)

        # Residuals and Fitted values
        residuals = results.resids
        fitted = results.predict()
        residuals.name = 'resid'
        fitted.name = 'fitted'

        # Merge with original df to ensure entity and time are preserved
        plot_df = df.copy()
        plot_df['resid'] = residuals
        plot_df['fitted'] = fitted
        plot_df = plot_df.reset_index()

        # Plot residuals vs fitted for each entity
        unique_entities = plot_df[self.entity].unique()
        n_cols = 5
        n_rows = math.ceil(len(unique_entities) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, entity in enumerate(unique_entities):
            subset = plot_df[plot_df[self.entity] == entity]
            axes[i].scatter(subset['fitted'], subset['resid'], alpha=0.6)
            axes[i].axhline(0, color='gray', linestyle='--')
            axes[i].set_title(f'{entity}')
            axes[i].set_xlabel("Fitted")
            axes[i].set_ylabel("Residuals")

        # Remove any unused subplots
        for ax in axes[len(unique_entities):]:
            fig.delaxes(ax)

        fig.suptitle('Residuals vs Fitted by Entity', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return results