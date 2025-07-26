import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# 1.Check VIF among features
def calculate_vif(df, exclude_columns=None):
    if exclude_columns is not None:
        df = df.drop(columns=exclude_columns, errors='ignore')

    df = df.astype(np.float64)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return vif_data.sort_values(by="VIF", ascending=False)


# 2.Mutual Information
def calculate_mutual_information(X, y, exclude_columns=None, random_state=42):
    # Drop specified columns from X if any
    if exclude_columns is not None:
        X = X.drop(columns=exclude_columns, errors='ignore')
    
    # Calculate mutual information scores
    mi = mutual_info_regression(X, y, random_state=random_state)
    
    # Sort features by MI score in descending order
    mi_scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
    
    return mi_scores

# 3.PCA reduction
def pca_reduction(df, features, n_components=2, prefix='pc', groupname=None):
    """
    Apply PCA to selected features and replace them in the original DataFrame.

    Parameters:
    - df: original DataFrame
    - features: list of column names to apply PCA to
    - n_components: number of principal components to keep
    - prefix: prefix for PCA component names (default='PC')
    - groupname: suffix to append for identifying the feature group, string 
    
    Returns:
    - df_new: DataFrame with original features dropped and PCA components added
    """
    # Step 1: Standardize selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Step 2: Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: Create new column names like PC1_CloseGroup, PC2_CloseGroup, ...
    pca_columns = [f"{prefix}{i+1}_{groupname}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)

    # Step 4: Drop original features and concatenate PCA results
    df_new = df.drop(columns=features).copy()
    df_new = pd.concat([df_new, pca_df], axis=1)

    return df_new


# 4.RandomForest Feature Selection
def feature_importance(
    X, y,
    exclude_columns=None,
    n_splits=5,
    n_estimators=2000,
    last_n_highlight=3,
    random_state=42
):
    """
    Calculate and plot feature importances using Random Forest with TimeSeriesSplit.

    Parameters:
    - X: Feature DataFrame
    - y: Target Series
    - exclude_columns: list of column names to exclude from X
    - n_splits: Number of splits for TimeSeriesSplit (default=5)
    - n_estimators: Number of trees in the Random Forest (default=2000)
    - top_n_highlight: Number of top features to highlight in a different color
    - random_state: Random seed

    Returns:
    - feature_importance_df: DataFrame sorted by importance
    """

    # Remove excluded columns
    if exclude_columns is not None:
        X = X.drop(columns=exclude_columns, errors='ignore')

    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances_list = []

    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train, y_train)
        importances_list.append(rf.feature_importances_)

    # Mean importance
    mean_importances = np.mean(importances_list, axis=0)

    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": mean_importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Plot
    colors = ['#1f77b4'] * (len(feature_importance_df) - last_n_highlight) + ['#ff7f0e'] * last_n_highlight
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color=colors)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances (TimeSeriesSplit)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return feature_importance_df


if __name__ == "__main__":
    df = pd.read_csv("feature_engineered.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.set_index('Date', inplace=True)
    print(df.columns)
    # print(df.isnull().sum()) 
    vif_result = calculate_vif(df, exclude_columns=['y1_30_lagged_ma30_pct_change', 'y2_1_lagged_pct_change'])
    print(vif_result)
    y1 = df['y1_30_lagged_ma30_pct_change']
    y2 = df['y2_1_lagged_pct_change']
    X = df.drop(columns=['y1_30_lagged_ma30_pct_change', 'y2_1_lagged_pct_change'])
    mi_1 = calculate_mutual_information(X, y1)
    mi_2 = calculate_mutual_information(X, y2)
    print(mi_1)
    print('--------------------')
    print(mi_2)
    # Based on MI and VIF discard features then selected features are below:
    selected_features_and_y = ['ma30_btc_close', 'ma30_fracdiff_btc_close', 'ma30_btc_volume', 'ma30_btc_volume_pct_change',
                             'ma30_btc_close_log_return', 'ma30_hash_rate', 'eth_volume','ma30_eth_close',
                             'ma30_eth_close_pct_change', 'xrp_volume', 'ma30_xrp_close', 'ma30_xrp_close_pct_change',
                            'sp_volume', 'ma30_sp_close', 'ma30_sp_close_pct_change', 'y1_30_lagged_ma30_pct_change']
    
    print(calculate_vif(df[selected_features_and_y], exclude_columns='y1_30_lagged_ma30_pct_change'))
    df1 = df[selected_features_and_y]
    df2 = pca_reduction(df1, features = ['ma30_btc_close', 'ma30_sp_close', 'ma30_eth_close'], n_components=2, prefix='pc', groupname='price')
    df3 = pca_reduction(df2, features = ['ma30_btc_volume', 'sp_volume', 'eth_volume'], n_components=2, prefix='pc', groupname='volume')
    print(calculate_vif(df3, exclude_columns='y1_30_lagged_ma30_pct_change'))
    X = df3.drop(columns=['y1_30_lagged_ma30_pct_change'])
    print(calculate_mutual_information(X, y1))
    # importance_ranked_df1 = feature_importance(X, y1, exclude_columns='ma30_hash_rate')
    # importance_ranked_df2 = feature_importance(X, y1, exclude_columns='ma30_fracdiff_btc_close')
    # print(importance_ranked_df1)
    # print(importance_ranked_df2)
    
    # Based on Randomforest selection discard last three features, remember have to make sure VIF under 10, 
    # otherwise importance ranking might be distorted
    final_df = df3.drop(columns=['pc2_volume', 'ma30_btc_volume_pct_change', 'xrp_volume'])
    print(final_df.columns)
    final_df.to_csv('selected_features_and_y.csv', index=True)



    # print(df.shape)