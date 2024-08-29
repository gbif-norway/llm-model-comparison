import pandas as pd
import json
from fuzzywuzzy import fuzz
import os
import plotly.express as px
import csv


def get_models_similarity_scores():
    df = pd.read_csv('data/TESTdata_subset_with_model_responses.csv', on_bad_lines='skip', dtype=str)

    models = ['llama-3.1-8b', 'gpt-4', 'gpt-4o-mini', 'gpt-4-turbo']
    model_similarities = {model: [] for model in models}
    for model in models:
        for _, row in df.iterrows():
            try:
                model_response = json.loads(row[f'{model}_response'], strict=False)
                print(model_response)
            except AttributeError:
                model_response = row[f'{model}_response'] 
            except json.JSONDecodeError:
                continue
            fuzzes = {}
            for col, val in model_response.items():
                if col in row:
                    human_val = row[col]
                    fuzzes[col] = fuzz.ratio(str(human_val), str(val))
            model_similarities[model].append(fuzzes)

    similarity_dfs = {model: pd.DataFrame.from_records(records) for model, records in model_similarities.items()}
    return similarity_dfs

def calculate_model_similarities(df, base_model, other_models):
    similarities = {}
    for other_model in other_models:
        model_similarities = []
        for _, row in df.iterrows():
            try:
                base_response = json.loads(row[f'{base_model}_response'], strict=False)
                other_response = json.loads(row[f'{other_model}_response'], strict=False)
            except (AttributeError, json.JSONDecodeError):
                continue
            
            fuzzes = {}
            for col in base_response.keys():
                if col in other_response:
                    fuzzes[col] = fuzz.ratio(str(base_response[col]), str(other_response[col]))
            model_similarities.append(fuzzes)
        
        similarities[other_model] = pd.DataFrame.from_records(model_similarities)
    
    return similarities

def plot_similarity_heatmap(similarity_df, title):
    model_means = []

    for model, df in similarity_df.items():
        column_means = df.mean()
        mean_total = column_means.mean()
        # Replace append with concat
        model_means.append(pd.concat([column_means, pd.Series({'mean_total': mean_total})], axis=0))

    rankings_table = pd.concat(model_means, axis=1).T
    rankings_table.index = similarity_df.keys()  # Set the index to model names
    rankings_table.index.name = 'Model'
    rankings_table.columns = [f"{col}_similarity_score_mean" for col in rankings_table.columns]

    # Create a heatmap using Plotly
    fig = px.imshow(rankings_table, 
                    labels=dict(x="", y="Model", color="Mean Similarity Score (0-100)"),
                    x=rankings_table.columns,
                    y=rankings_table.index,
                    aspect="auto",
                    color_continuous_scale=[(0, "white"), (1, "blue")])  # White to blue color scale
    fig.update_layout(title=title,
                      xaxis_title="Scores range from 0 (different) to 100 (identical)",
                      yaxis_title="Model",
                      coloraxis_colorbar=dict(title="Mean Similarity Score (0-100)"))
    
    return fig

# Main execution
models = ['llama-3.1-8b', 'gpt-4', 'gpt-4o-mini', 'gpt-4-turbo']
df = pd.read_csv('data/TESTdata_subset_with_model_responses.csv', on_bad_lines='skip', dtype=str)

# Create the 'comparisons' folder if it doesn't exist
os.makedirs('comparisons', exist_ok=True)

# Human vs Models heatmap
model_similarity_scores = get_models_similarity_scores()
fig = plot_similarity_heatmap(model_similarity_scores, "Human vs Models Similarity Scores")
fig.write_html("comparisons/human_vs_models_similarity_heatmap.html")

# Model vs Model heatmaps
for base_model in models:
    other_models = [model for model in models if model != base_model]
    similarity_dict = calculate_model_similarities(df, base_model, other_models)
    fig = plot_similarity_heatmap(similarity_dict, f"{base_model} vs Other Models Similarity Scores")
    fig.write_html(f"comparisons/{base_model}_vs_others_similarity_heatmap.html")