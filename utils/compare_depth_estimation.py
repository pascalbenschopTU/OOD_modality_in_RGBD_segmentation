import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim
from skimage import filters
import multiprocessing as mp
import optuna


# Helper function to calculate KL Divergence
def calculate_kl_divergence(p, q, bins=100):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    kl_div = entropy(p_hist + 1e-6, q_hist + 1e-6)  # Add small constant to avoid log(0)
    return kl_div

# Helper function to calculate Sobel gradients for depth images
def calculate_gradient(image):
    sobel_x = filters.sobel_h(image)
    sobel_y = filters.sobel_v(image)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient_magnitude

# Resize both images to the smallest size between them
def resize_to_minimum(image1, image2):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    # Find the minimum dimensions between the two images
    new_height = min(height1, height2)
    new_width = min(width1, width2)

    # Resize both images to the new dimensions
    resized_image1 = cv2.resize(image1, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_image2 = cv2.resize(image2, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image1, resized_image2

# Function to apply augmentations to depth images
def apply_augmentations(depth_images, occlusion_prob=0.1, noise_std=0.01):
    augmented_images = []
    
    print("Applying augmentations...")
    for depth_image in tqdm(depth_images):
        # Randomly apply zero-depth occlusions
        if np.random.rand() < occlusion_prob:
            mask = np.random.rand(*depth_image.shape) < 0.1  # Randomly occlude 10% of pixels
            depth_image[mask] = 0
        
        # Add random Gaussian noise
        noise = np.random.normal(0, noise_std, depth_image.shape)
        depth_image = np.clip(depth_image + noise, 0, 1)

        depth_image = depth_image.astype(np.float32)
        
        augmented_images.append(depth_image)
    
    return augmented_images

def process_image(depth_image):
    depth_values = depth_image.flatten()
    gradient = calculate_gradient(depth_image).flatten()

    return depth_values, gradient


def aggregate_metrics(depth_images):
    print("Starting aggregation for a large number of images...")

    # Use multiprocessing for parallel processing of images
    pool = mp.Pool(mp.cpu_count())  # Use all available cores
    results = pool.map(process_image, depth_images)  # Process images in parallel
    pool.close()  # Close the pool after processing
    pool.join()  # Wait for all processes to finish

    # Separate depth values and gradients from results
    all_depth_values, all_gradients = zip(*results)
    
    # Concatenate depth values and gradients
    all_depth_values = np.concatenate(all_depth_values)
    all_gradients = np.concatenate(all_gradients)
    print("Processed and concatenated depth values and gradients")

    # Calculate statistics
    mean_depth = np.mean(all_depth_values)
    std_depth = np.std(all_depth_values)
    min_depth = np.min(all_depth_values)
    max_depth = np.max(all_depth_values)
    print("Calculated statistics")

    # Compute histogram
    depth_hist, depth_bins = np.histogram(all_depth_values, bins=100, density=True)
    print("Computed histogram")
    
    return mean_depth, std_depth, min_depth, max_depth, depth_hist, depth_bins, all_gradients


# def update_augmentations_based_on_metrics(night_stats, depth_images_daytime, original_day_stats):
#     # Unpack stats for nighttime dataset
#     _, _, _, _, depth_hist_night, depth_bins_night, _ = night_stats
#     _, _, _, _, depth_hist_day_original, depth_bins_day_original, _ = original_day_stats

#     # Set initial augmentation parameters
#     occlusion_prob = 0.1
#     noise_std = 0.01

#     best_kl_divergence = 1000  # Initialize with a high value

#     for iteration in range(10):  # Perform 10 iterations
#         print(f"\nIteration {iteration + 1}")

#         # Apply augmentations to daytime images
#         augmented_images = apply_augmentations(
#             depth_images_daytime,
#             occlusion_prob=occlusion_prob,
#             noise_std=noise_std,
#         )

#         # Aggregate metrics for augmented daytime images
#         day_stats = aggregate_metrics(augmented_images)
#         _, _, _, _, depth_hist_day, depth_bins_day, _ = day_stats
        
#         print("\n--- Depth Statistics Comparison ---")

#         # Calculate KL Divergence between augmented daytime and nighttime distributions
#         kl_divergence_depth = calculate_kl_divergence(depth_hist_day, depth_hist_night)
#         print(f"KL Divergence (Depth Distribution): {kl_divergence_depth:.4f}")

#         # Visualize distribution comparison
#         plt.figure(figsize=(10, 6))
#         plt.title(f"Iteration {iteration + 1}: Depth Value Distribution: Daytime (Augmented) vs Nighttime")
#         plt.plot(depth_bins_day[:-1], depth_hist_day, label="Daytime (Augmented)", color='blue')
#         plt.plot(depth_bins_night[:-1], depth_hist_night, label="Nighttime", color='orange')
#         plt.plot(depth_bins_day_original[:-1], depth_hist_day_original, label="Daytime (Original)", color='green', linestyle='--')
#         plt.xlabel("Depth Value")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.show()

#         # Save best augmentation parameters based on KL Divergence
#         if iteration == 0 or kl_divergence_depth < best_kl_divergence:
#             best_kl_divergence = kl_divergence_depth
#             best_params = (occlusion_prob, noise_std)
#             print("Best KL Divergence updated!")
#             with open("best_augmentation_params.txt", "a") as f:
#                 f.write(f"Best KL Divergence: {best_kl_divergence:.4f}\n")
#                 f.write(f"Best Parameters: {best_params}\n")

#         # Update augmentation parameters for the next iteration
#         occlusion_prob += 0.05  # Increase occlusion probability
#         noise_std += 0.002  # Increase noise


def update_augmentations_with_optuna(night_stats, depth_images_daytime, original_day_stats):
    # Unpack stats for nighttime dataset
    _, _, _, _, depth_hist_night, depth_bins_night, _ = night_stats
    _, _, _, _, depth_hist_day_original, depth_bins_day_original, _ = original_day_stats

    def objective(trial):
        # Suggest values for occlusion_prob and noise_std using Optuna
        occlusion_prob = trial.suggest_float('occlusion_prob', 0.0, 0.2)
        noise_std = trial.suggest_float('noise_std', 0.0, 0.1)

        # Apply augmentations with the suggested parameters
        augmented_images = apply_augmentations(
            depth_images_daytime,
            occlusion_prob=occlusion_prob,
            noise_std=noise_std,
        )

        # Aggregate metrics for augmented daytime images
        day_stats = aggregate_metrics(augmented_images)
        _, _, _, _, depth_hist_day, depth_bins_day, _ = day_stats

        # Calculate KL Divergence between augmented daytime and nighttime distributions
        kl_divergence_depth = calculate_kl_divergence(depth_hist_day, depth_hist_night)
        print(f"KL Divergence (Depth Distribution): {kl_divergence_depth:.4f}")

        return kl_divergence_depth  # This is the loss to minimize

    # Create an Optuna study and start optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)  # You can adjust the number of trials

    # Output the best parameters and KL divergence
    print(f"Best KL Divergence: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")

    # Unpack best parameters
    occlusion_prob = study.best_params['occlusion_prob']
    noise_std = study.best_params['noise_std']
    contrast_factor = study.best_params['contrast_factor']

    # Apply augmentations with the best parameters
    augmented_images = apply_augmentations(
        depth_images_daytime,
        occlusion_prob=occlusion_prob,
        noise_std=noise_std,
    )

    # Aggregate metrics for augmented daytime images
    day_stats = aggregate_metrics(augmented_images)
    _, _, _, _, depth_hist_day, depth_bins_day, _ = day_stats

    # Plot for visualization (optional, can comment out if slowing down trials)
    plt.figure(figsize=(10, 6))
    plt.title(f"Depth Value Distribution: Daytime (Augmented) vs Nighttime")
    plt.plot(depth_bins_day[:-1], depth_hist_day, label="Daytime (Augmented)", color='blue')
    plt.plot(depth_bins_night[:-1], depth_hist_night, label="Nighttime", color='orange')
    plt.plot(depth_bins_day_original[:-1], depth_hist_day_original, label="Daytime (Original)", color='green', linestyle='--')
    plt.xlabel("Depth Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Optionally save the best parameters to a file
    with open("best_augmentation_params.txt", "a") as f:
        f.write(f"Best KL Divergence: {study.best_value:.4f}\n")
        f.write(f"Best Parameters: {study.best_params}\n")

    return study.best_params, study.best_value


# Main function to parse dataset locations and process files
def main():
    parser = argparse.ArgumentParser(description="Compare depth image distributions between two datasets")
    parser.add_argument("-d1", "--dataset_location_1", type=str, required=True, help="Location of the first dataset (daytime)")
    parser.add_argument("-d2", "--dataset_location_2", type=str, required=True, help="Location of the second dataset (nighttime)")
    args = parser.parse_args()

    dataset_location_1 = args.dataset_location_1
    dataset_location_2 = args.dataset_location_2

    # Get list of files from both datasets
    files_1 = [os.path.join(root, file) for root, _, files in os.walk(dataset_location_1) for file in files if file.endswith('.png')]
    files_2 = [os.path.join(root, file) for root, _, files in os.walk(dataset_location_2) for file in files if file.endswith('.png')]

    # Limit to 500 files each
    files_1 = files_1[:500]
    files_2 = files_2[:500]

    # Load all depth images from both datasets
    depth_images_daytime = []
    depth_images_nighttime = []
    
    print("Loading daytime depth images...")
    for file in tqdm(files_1):
        depth_image = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_image /= np.max(depth_image)  # Normalize depth image
        depth_images_daytime.append(depth_image)

    print("Loading nighttime depth images...")
    for file in tqdm(files_2):
        depth_image = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_image /= np.max(depth_image)  # Normalize depth image
        depth_images_nighttime.append(depth_image)

    # Calculate aggregate metrics for both datasets
    day_stats = aggregate_metrics(depth_images_daytime)
    night_stats = aggregate_metrics(depth_images_nighttime)

    # Compare the overall statistics and distributions between the datasets
    # compare_distributions(day_stats, night_stats)

    # Update augmentation parameters based on nighttime dataset metrics
    update_augmentations_with_optuna(night_stats, depth_images_daytime, day_stats)

if __name__ == "__main__":
    main()
