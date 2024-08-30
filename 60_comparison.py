import pandas as pd
import json
from fuzzywuzzy import fuzz
import os
import plotly.express as px
import csv
import plotly.graph_objects as go


def get_models_similarity_scores():
    df = pd.read_csv('data/data_subset_with_model_responses.csv', on_bad_lines='skip', dtype=str)
    df = df[~df[['llama-3.1-8b_response', 'gpt-4o_response', 'gpt-4_response', 'gpt-4o-mini_response', 'gpt-4-turbo_response', 'claude-3.5-sonnet_response']].isin(['FAIL', '{}', '']).any(axis=1)]
    df = df[df[['llama-3.1-8b_response', 'gpt-4o_response', 'gpt-4_response', 'gpt-4o-mini_response', 'gpt-4-turbo_response', 'claude-3.5-sonnet_response']].apply(lambda x: x.str.strip() != '{}').all(axis=1)]

    models = ['llama-3.1-8b', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'claude-3.5-sonnet']
    model_similarities = {model: [] for model in models}
    for model in models:
        for _, row in df.iterrows():
            try:
                model_response = row[f'{model}_response']
                if model_response.startswith('```json'):
                    model_response = model_response[len("```json"):]
                if model_response.endswith("```"):
                    model_response = model_response[:-len("```")]
                model_response = json.loads(model_response, strict=False)
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

def plot_similarity_heatmap(similarity_df, title, num_records):
    model_means = []

    for model, df in similarity_df.items():
        column_means = df.mean()
        mean_total = column_means.mean()
        model_means.append(pd.concat([pd.Series({'mean_total': mean_total}), column_means], axis=0))

    rankings_table = pd.concat(model_means, axis=1).T
    rankings_table.index = similarity_df.keys()  # Set the index to model names
    rankings_table.index.name = 'Model'
    
    # Remove the "_similarity_score_mean" suffix from column names
    rankings_table.columns = [col.replace('_similarity_score_mean', '') for col in rankings_table.columns]

    # Ensure mean_total is the first column
    cols = rankings_table.columns.tolist()
    mean_total_col = 'mean_total'
    cols.remove(mean_total_col)
    cols = [mean_total_col] + cols
    rankings_table = rankings_table[cols]

    # Sort the rankings_table by the mean_total column in descending order
    rankings_table = rankings_table.sort_values(mean_total_col, ascending=False)

    # Create a heatmap using Plotly
    fig = go.Figure()

    # Add the mean total column (in red)
    fig.add_trace(go.Heatmap(
        z=rankings_table[mean_total_col].values.reshape(-1, 1),
        x=[mean_total_col],
        y=rankings_table.index,
        colorscale=[(0, "white"), (1, "red")],
        showscale=False,
        name="Mean Total",
        hovertemplate='Model: %{y}<br>%{x}: %{z:.2f}<extra></extra>'
    ))

    # Add the rest of the columns (in blue)
    fig.add_trace(go.Heatmap(
        z=rankings_table[cols[1:]].values,
        x=cols[1:],
        y=rankings_table.index,
        colorscale=[(0, "white"), (1, "blue")],
        showscale=True,
        colorbar=dict(title="Mean Similarity Score (0-100)"),
        name="Individual Scores",
        hovertemplate='Model: %{y}<br>%{x}: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{title} (n={num_records})",
        xaxis_title="Scores range from 0 (different) to 100 (identical)",
        yaxis_title="Model",
    )

    # Make the first column wider
    fig.update_xaxes(type='category')
    fig.update_layout(xaxis=dict(
        tickson="boundaries",
        ticklen=10,
        showgrid=True,
        gridwidth=2,
        gridcolor="LightGrey",
    ))
    
    return fig

# Main execution
models = ['llama-3.1-8b', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'claude-3.5-sonnet']
df = pd.read_csv('data/data_subset_with_model_responses.csv', on_bad_lines='skip', dtype=str)
df = df[~df[['llama-3.1-8b_response', 'gpt-4o_response', 'gpt-4_response', 'gpt-4o-mini_response', 'gpt-4-turbo_response', 'claude-3.5-sonnet_response']].isin(['FAIL', '{}', '']).any(axis=1)]
df = df[df[['llama-3.1-8b_response', 'gpt-4o_response', 'gpt-4_response', 'gpt-4o-mini_response', 'gpt-4-turbo_response', 'claude-3.5-sonnet_response']].apply(lambda x: x.str.strip() != '{}').all(axis=1)]

num_records = len(df)

# Create the 'comparisons' folder if it doesn't exist
os.makedirs('comparisons', exist_ok=True)

# Human vs Models heatmap
model_similarity_scores = get_models_similarity_scores()
fig = plot_similarity_heatmap(model_similarity_scores, "Human vs Models Similarity Score Means Across Columns", num_records)
fig.write_html("comparisons/human_vs_models_similarity_heatmap.html")

# Model vs Model heatmaps
for base_model in models:
    other_models = [model for model in models if model != base_model]
    similarity_dict = calculate_model_similarities(df, base_model, other_models)
    fig = plot_similarity_heatmap(similarity_dict, f"{base_model} vs Other Models Similarity Score Means Across Columns", num_records)
    fig.write_html(f"comparisons/{base_model}_vs_others_similarity_heatmap.html")