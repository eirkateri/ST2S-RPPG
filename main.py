import os
import cv2
import glob
import yaml
import torch
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import other modules
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from st_cropping import CropVideos
from st_image_generator import SpatioTemporal
from create_dataset import CustomImageDataset
from train import Loss
from CNN import CnnRegressor
from prediction_only import CnnClassifier
from result_analysis import analyze_results
from binary_class_generation import BinaryClassGenerator
from classifier import run_mlp_experiments

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# the video stabilizing models can be found here: https://github.com/aharley/pips/tree/main (we are not the authors of this)

def preprocess_videos():
    source_folder = config['data_paths']['source_folder']
    cropped_folder = config['data_paths']['cropped_folder']
    os.makedirs(cropped_folder, exist_ok=True)

    for file in tqdm(glob.glob(os.path.join(source_folder, '*.mp4'))):
        CropVideos(cropped_folder, source_folder, file, config['video_processing']['pyramid_levels']).crop_video(config['video_processing']['plot'])

def generate_spatio_temporal_images():
    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Retrieve parameters from the config file
    end_folder = config['data_paths']['generated_data_folder']
    source_folder = config['data_paths']['source_folder']
    pixel_width = config['video_processing']['pixel_width']

    # Ensure the output directory exists
    os.makedirs(end_folder, exist_ok=True)

    # Automatically get list of video files from the source folder
    video_files = [os.path.join(source_folder, file) for file in os.listdir(source_folder) if file.endswith('.mp4')]

    # Generate spatio-temporal images for each video
    for file_name in video_files:
        spatio_temporal = SpatioTemporal(end_folder, file_name, pixel_width)
        spatio_temporal.st_image()


def crop_images_to_minimum_size():
    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set directory paths from config
    cropped_folder = config['data_paths']['cropped_folder']
    source_folder = config['data_paths']['source_folder']

    os.chdir(cropped_folder)

    # Initialize min height and width
    min_height, min_width = 10000, 10000

    # Find minimum dimensions
    for image in tqdm(glob.glob('*.jpg')):
        img = cv2.imread(image)
        min_height = min(min_height, img.shape[0])
        min_width = min(min_width, img.shape[1])

    print(f"Minimum height: {min_height}, Minimum width: {min_width}")

    # Crop all images to the minimum dimensions
    for image in tqdm(glob.glob('*.jpg')):
        img = cv2.imread(image)
        crop_img = img[0:min_height, 0:min_width]
        cv2.imwrite(os.path.join(source_folder, image), crop_img)

    print(f"All images cropped to {min_height}x{min_width} and saved to {source_folder}.")


def train_model():
    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set the random seed for reproducibility
    rand_seed = config['training']['seed']
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    model_data = config['data_paths']['model_data']
    target_csv = config['data_paths']['target_csv']

    # Load the full dataset using the CustomImageDataset class
    target = pd.read_csv(target_csv)
    target['SUBJECT'] = target['ID'].astype(str) + '_' + target['segment'].astype(str)
    dataset = CustomImageDataset(dataframe=target, image_folder=model_data)

    # Extract the subject to leave out for testing
    lopo_subject = config['training']['lopo']  # Subject to leave out

    # Identify indices for the train and test sets based on the subject ID
    train_indices = target[~target['SUBJECT'].str.contains(lopo_subject)].index.tolist()
    test_indices = target[target['SUBJECT'].str.contains(lopo_subject)].index.tolist()

    # Create train and test datasets using Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Convert datasets to DataLoader
    batch_size = config['training']['batch_size']
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize the model
    model = CnnRegressor(batch_size, 3, 1).cuda()
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    epochs = config['training']['epochs']

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        avg_loss, avg_r2_score, _, _, _, _ = Loss(model, loader_train, train=True, optimizer=optimizer).model_loss()
        print(f"Epoch {epoch + 1}:\n\tLoss = {avg_loss}\n\tR^2 Score = {avg_r2_score}")
        train_losses.append(avg_loss)

    # Save the trained model
    model_save_path = os.path.join(config['data_paths']['model_save_path'], f"{lopo_subject}_model.h")
    torch.save(model.state_dict(), model_save_path)

    # Plot and save the loss curve
    plt.plot(train_losses)
    plt.savefig(os.path.join(config['data_paths']['model_save_path'], f"{lopo_subject}_lossplot.png"))

    # Model evaluation
    avg_loss, avg_r2_score, pred, out, all_pred, all_out = Loss(model, loader_test).model_loss()
    res = pd.DataFrame({
        'Subject': target.loc[test_indices, 'SUBJECT'],
        'Predictions': all_pred,
        'Output': all_out
    })
    res['Difference'] = abs(res['Predictions'] - res['Output'])
    res.to_csv(os.path.join(config['data_paths']['results_folder'], f"{lopo_subject}_results.csv"), index=False)

    print(f"The model's L1 loss is: {avg_loss}")
    print(f"The model's R^2 score is: {avg_r2_score}")

def evaluate_model():
    analyze_results()

def run_inference():
    pretrained_model = torch.load(config['data_paths']['model_data'] + "/model.h")
    model = CnnClassifier(pretrained_model)
    model.eval()

    df = pd.read_csv(config['data_paths']['target_csv'])
    test_loader = DataLoader(df, batch_size=config['training']['batch_size'], shuffle=False)

    results = []
    for inputs, targets in tqdm(test_loader):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        results.extend(preds.cpu().numpy())

    result_df = pd.DataFrame({"Predicted Class": results})
    result_df.to_csv(config['output']['results_csv_path'], index=False)

def generate_binary_dataset():
    generator = BinaryClassGenerator()
    generator.generate_binary_classes()

def run_mlp():
    run_mlp_experiments()

def main():
    preprocess_videos()
    generate_spatio_temporal_images()
    crop_images_to_minimum_size()
    train_model()
    evaluate_model()
    run_inference()
    generate_binary_dataset()
    run_mlp_experiments()


if __name__ == "__main__":
    main()