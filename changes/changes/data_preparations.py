#!/usr/bin/env python
# data_preparations.py
# coding: utf-8
#Converted to .py


import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_offline_data(e1_path, e2_path, output_path):
    e1 = pd.read_csv(e1_path)
    e2 = pd.read_csv(e2_path)
    
    e1['id'] = e1['id'].str.replace('e1det_', '', regex=False)
    e2['id'] = e2['id'].str.replace('e2det_', '', regex=False)
    
    data = e1.merge(e2, on=['begin', 'end', 'id'], how='inner')
    
    numeric_data = data.drop(columns=["id", "begin", "end"], errors='ignore')
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)

    pca = PCA()
    pca_transformed = pca.fit_transform(standardized_data)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    n_components_95 = (cumulative_variance >= 0.95).argmax() + 1
    selected_components = pca.components_[:n_components_95]
    selected_features = numeric_data.columns.tolist()
    selected_components_df = pd.DataFrame(selected_components, columns=selected_features)

    absolute_weights_sum = selected_components_df.abs().sum(axis=0)
    ranked_features = absolute_weights_sum.sort_values(ascending=False)

    top_features = ranked_features.index[1:n_components_95].tolist()

    essential_features = ['flow']
    final_features = list(set(top_features + essential_features))

    data['hour'] = ((data['begin'] // 3600) + 1).astype(int)
    
    data.loc[:, 'group'] = data['id'].apply(lambda x: x.split('_')[0])
    group_mapping = {
        '412441780#1': 'South',
        '688169644#0': 'East',
        '688171462#2': 'West',
        '688214484#0': 'North'
    }
    data['group_label'] = data['group'].map(group_mapping)

    data['target'] = data.groupby(['group', 'hour'])['occupancy'].transform('mean')

    keep_cols = ['id', 'group', 'group_label', 'hour', 'target'] + final_features
    data = data[keep_cols].copy()

    data.sort_values(by=['group', 'hour'], inplace=True)
    numeric = data.drop(columns=['id', 'group', 'group_label', 'hour'], errors='ignore')
    scaler = MinMaxScaler()
    data[numeric.columns] = scaler.fit_transform(numeric)
    order = ['hour', 'id', 'group_label', 'group', 'flow', 'occupancy', 'speed', 'meanMaxJamLengthInMeters', 'meanHaltingDuration', 'meanSpeed', 'target']
    data = data[order]

    data.to_csv(output_path, index=False)
    print(f"Final data saved to: {output_path}")

if __name__ == "__main__":
    e1_path = r'e1_output.csv'
    e2_path = r'e2_output.csv'
    output_path = r'final_data.csv'
    prepare_offline_data(e1_path, e2_path, output_path)

