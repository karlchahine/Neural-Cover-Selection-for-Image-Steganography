
import pandas as pd
from brisque import BRISQUE
from PIL import Image
import os
import numpy as np
import argparse

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--bpp', type=int, default=3, help='Payload: number of bits to be hidden per pixel')
    parser.add_argument('--dataset-class', type=str, default="CelebAHQ", help="Class of dataset")
 
    args = parser.parse_args()

    return args


args = parse_args_and_config()

# Path to the CSV file
csv_path = args.dataset_class + '_' + str(args.bpp) + 'bpp.csv'

# Path to the folder containing the images
image_folder_path = "./Stego_images/" + args.dataset_class + "/" + str(args.bpp) + "bpp"

# Read the CSV file
data = pd.read_csv(csv_path)

# Group the data by 'index' and find the iteration with the lowest error for each index
best_iterations = data.loc[data.groupby('index')['error'].idxmin()]
obj = BRISQUE(url=False)
# Initialize lists to store metrics
brisque_scores = []
errors = []
psnrs = []
ssims = []

# Loop through each row in the filtered dataframe
for _, row in best_iterations.iterrows():
    image_name = f"image{int(row['index'])}_{int(row['iteration'])}.png"
    image_path = os.path.join(image_folder_path, image_name)
    
    # Check if the image exists
    if os.path.exists(image_path):
        # Open the image and calculate the BRISQUE score
        img = Image.open(image_path)
        ndarray = np.asarray(img)
        brisque_score = obj.score(img=ndarray)

        # Append the BRISQUE score and other metrics to the lists
        brisque_scores.append(brisque_score)
        errors.append(row['error'])
        psnrs.append(row['psnr'])
        ssims.append(row['ssim'])
    else:
        print(f"Image not found: {image_path}")

# Calculate the averages
average_brisque = sum(brisque_scores) / len(brisque_scores) if brisque_scores else 0
average_error = sum(errors) / len(errors) if errors else 0
average_psnr = sum(psnrs) / len(psnrs) if psnrs else 0
average_ssim = sum(ssims) / len(ssims) if ssims else 0

# Print the averages
print(f"Average BRISQUE Score: {average_brisque}")
print(f"Average Error: {average_error}")
print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")


# Filter data to get only iteration 0 for all indexes
iteration_zero_data = data[data['iteration'] == 0]

# Initialize lists to store metrics for iteration 0
brisque_scores_0 = []
errors_0 = []
psnrs_0 = []
ssims_0 = []

# Loop through each row in the iteration zero dataframe
for _, row in iteration_zero_data.iterrows():
    image_name = f"image{int(row['index'])}_{int(row['iteration'])}.png"
    image_path = os.path.join(image_folder_path, image_name)
    
    # Check if the image exists
    if os.path.exists(image_path):
        # Open the image and calculate the BRISQUE score
        img = Image.open(image_path)
        ndarray = np.asarray(img)
        brisque_score = obj.score(img=ndarray)
        
        # Append the BRISQUE score and other metrics to the lists for iteration 0
        brisque_scores_0.append(brisque_score)
        errors_0.append(row['error'])
        psnrs_0.append(row['psnr'])
        ssims_0.append(row['ssim'])
    else:
        print(f"Image not found: {image_path}")

# Calculate the averages for iteration 0
average_brisque_0 = sum(brisque_scores_0) / len(brisque_scores_0) if brisque_scores_0 else 0
average_error_0 = sum(errors_0) / len(errors_0) if errors_0 else 0
average_psnr_0 = sum(psnrs_0) / len(psnrs_0) if psnrs_0 else 0
average_ssim_0 = sum(ssims_0) / len(ssims_0) if ssims_0 else 0

# Print the averages for iteration 0
print(f"Average BRISQUE Score for Iteration 0: {average_brisque_0}")
print(f"Average Error for Iteration 0: {average_error_0}")
print(f"Average PSNR for Iteration 0: {average_psnr_0}")
print(f"Average SSIM for Iteration 0: {average_ssim_0}")



