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

def plot_similarity_heatmap(model_similarity_scores):
    model_means = []

    for model, df in model_similarity_scores.items():
        column_means = df.mean()
        mean_total = column_means.mean()
        # Replace append with concat
        model_means.append(pd.concat([column_means, pd.Series({'mean_total': mean_total})], axis=0))

    rankings_table = pd.concat(model_means, axis=1).T
    rankings_table.index = model_similarity_scores.keys()  # Set the index to model names
    rankings_table.index.name = 'Model'
    rankings_table.columns = [f"{col}_similarity_score_mean" for col in rankings_table.columns]

    # Create a heatmap using Plotly
    fig = px.imshow(rankings_table, 
                    labels=dict(x="", y="Model", color="Mean Similarity Score (0-100)"),
                    x=rankings_table.columns,
                    y=rankings_table.index,
                    aspect="auto",
                    color_continuous_scale=[(0, "white"), (1, "blue")])  # White to blue color scale
    fig.update_layout(title="Model Similarity Scores Heatmap (Higher Score = More Similar)",
                      xaxis_title="Scores range from 0 (different) to 100 (identical)",
                      yaxis_title="Model",
                      coloraxis_colorbar=dict(title="Mean Similarity Score (0-100)"))
    
    return fig

# Assuming model_similarity_scores is defined as per your previous code
model_similarity_scores = get_models_similarity_scores()
fig = plot_similarity_heatmap(model_similarity_scores)
fig.write_html("similarity_heatmap.html")
import pdb; pdb.set_trace()
fig.show()