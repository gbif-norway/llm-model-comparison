#%%
!module load Python libffi bzip2 Pillow-SIMD

#%%
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

#%% Load the data subset
df = pd.read_csv('data/data_subset.csv')

#%% Function to check if URL is a valid image
def is_valid_image_url(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and response.headers.get('content-type', '').startswith('image/')
    except:
        return False

#%% Check URLs in parallel
with ThreadPoolExecutor(max_workers=10) as executor:
    df['is_valid_image'] = list(executor.map(is_valid_image_url, df['identifier']))

#%% Display results
print(df[['identifier', 'is_valid_image']].head())
print(f"Valid image URLs: {df['is_valid_image'].sum()} out of {len(df)}")
# %%
import os
from urllib.parse import urlparse
from tqdm import tqdm

# Create a directory to store the images
image_dir = 'data/images'
os.makedirs(image_dir, exist_ok=True)

# Function to download and save an image
def download_image(row):
    if row['is_valid_image']:
        url = row['identifier']
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Extract filename from URL
                filename = os.path.basename(urlparse(url).path)
                # Ensure unique filename by prepending the index
                unique_filename = f"{row.name}_{filename}"
                filepath = os.path.join(image_dir, unique_filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return filepath
        except:
            pass
    return None

# Download images and update DataFrame
print("Downloading images...")
tqdm.pandas()
df['local_path'] = df.progress_apply(download_image, axis=1)

# Display results
print(f"Downloaded {df['local_path'].notna().sum()} images out of {len(df)} valid URLs")

# Save updated DataFrame
df.to_csv('data/data_subset_with_local_paths.csv', index=False)
print("Updated DataFrame saved to 'data/data_subset_with_local_paths.csv'")

# %%
