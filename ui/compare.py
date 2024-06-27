import pandas as pd
from fuzzywuzzy import fuzz
from input.dwc_keys import fields
import os
import plotly.express as px

def compute_similarity_scores(human_df, model_df):
    human_df.set_index('gbifID', inplace=True)
    model_df.set_index('gbifID', inplace=True)

    if not human_df.index.equals(model_df.index):
        raise ValueError("Both DataFrames must have the same 'gbifID' indices.")
    
    similarity_scores_df = pd.DataFrame(index=human_df.index, columns=human_df.columns)
    for col in human_df.columns:
        for idx in human_df.index:
            human_text = str(human_df.at[idx, col])
            model_text = str(model_df.at[idx, col])
            similarity_scores_df.at[idx, col] = fuzz.ratio(human_text, model_text)
    
    return similarity_scores_df

def get_models_similarity_scores():
    human_df = pd.read_csv('input/data_subset.csv', usecols=list(fields.keys()))
    base_path = 'output/'

    models = {}
    for model_dir in os.listdir(base_path):
        model_path = os.path.join(base_path, model_dir)
        similarities = []

        # Get the mean across all the runs
        for file_name in os.listdir(model_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(model_path, file_name)
                model_df = pd.read_csv(file_path)
                similarities.append(compute_similarity_scores(human_df, model_df))
        models[model_dir] = pd.concat(similarities).groupby(level=0).mean()
    return models

def plot_similarity_heatmap(model_similarity_scores):
    model_means = []

    for model, df in model_similarity_scores.items():
        column_means = df.mean()
        mean_total = column_means.mean()
        model_means.append(column_means.append(pd.Series({'mean_total': mean_total}, name=model)))

    rankings_table = pd.concat(model_means, axis=1).T
    rankings_table.index.name = 'Model'
    rankings_table.columns = [f"{col}_similarity_score_mean" for col in rankings_table.columns]

    # Create a heatmap using Plotly
    fig = px.imshow(rankings_table, 
                    labels=dict(x="Metric", y="Model", color="Mean Similarity Score"),
                    x=rankings_table.columns,
                    y=rankings_table.index,
                    aspect="auto",
                    color_continuous_scale='Viridis')
    fig.update_layout(title="Model Similarity Scores Heatmap",
                      xaxis_title="Metric",
                      yaxis_title="Model",
                      coloraxis_colorbar=dict(title="Mean Similarity Score"))
    return fig

# Assuming model_similarity_scores is defined as per your previous code
model_similarity_scores = get_models_similarity_scores()
fig = plot_similarity_heatmap(model_similarity_scores)
fig.show()
