import os
import pandas as pd
from PIL import Image
import math

def downscale_image(input_path, output_path, max_size_mb=20):
    # Open the image
    with Image.open(input_path) as img:
        # Get the original size and format
        original_format = img.format
        width, height = img.size

        # Calculate the current file size in MB
        current_size_mb = os.path.getsize(input_path) / (1024 * 1024)

        # If the image is already smaller than max_size_mb, just save it
        if current_size_mb <= max_size_mb:
            img.save(output_path, format=original_format)
            return

        # Calculate the scaling factor
        scale_factor = math.sqrt(max_size_mb / current_size_mb)

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the resized image with maximum quality
        resized_img.save(output_path, format=original_format, quality=95)

# Read the CSV file
df = pd.read_csv('data/data_subset_with_local_paths.csv')

# Filter rows where is_valid_image is True
df = df[df['is_valid_image'] == True]

# Create the output directory if it doesn't exist
os.makedirs('./data/scaled', exist_ok=True)

# Add a new column for scaled image paths
df['scaled_local_path'] = ''

# Process each image
for index, row in df.iterrows():
    input_path = row['local_path']
    output_filename = os.path.basename(input_path)
    output_path = os.path.join('./data/scaled', output_filename)

    # Check if the scaled version already exists
    if os.path.exists(output_path):
        print(f"Scaled version already exists: {output_path}")
    else:
        print(f"Processing: {input_path}")
        downscale_image(input_path, output_path)
        print(f"Saved scaled image to: {output_path}")

    # Update the DataFrame with the scaled image path
    df.at[index, 'scaled_local_path'] = output_path

# Save the updated DataFrame
df.to_csv('data/data_subset_with_scaled_paths.csv', index=False)

print("All images have been processed and saved in ./data/scaled")
print("Updated CSV saved as data/data_subset_with_scaled_paths.csv")
