import cv2
import numpy as np
import matplotlib.pyplot as plt



city_seg_label_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\Cityscapes\gtFine\val\frankfurt\frankfurt_000000_008451_gtFine_color.png"
city_img_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\Cityscapes\leftImg8bit\val\frankfurt\frankfurt_000000_008451_leftImg8bit.png"
city_depth_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\Cityscapes\depth\val\frankfurt\frankfurt_000000_008451_depth.png"

nd_seg_label_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\NighttimeDrivingTest\gtCoarse_daytime_trainvaltest\test\night\0_frame_0205_gtCoarse_color.png"
nd_img_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\NighttimeDrivingTest\leftImg8bit\test\night\0_frame_0205_leftImg8bit.png"
nd_depth_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\NighttimeDrivingTest\depth\test\night\0_frame_0205_depth.png"


dz_seg_label_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\Dark_Zurich_val_anon\gt\val\night\GOPR0356\GOPR0356_frame_000345_gt_labelColor.png"
dz_img_filename =  r"G:\OOD_modality_in_RGBD_segmentation\datasets\Dark_Zurich_val_anon\rgb_anon\val\night\GOPR0356\GOPR0356_frame_000345_rgb_anon.png"
dz_depth_filename = r"G:\OOD_modality_in_RGBD_segmentation\datasets\Dark_Zurich_val_anon\depth\val\night\GOPR0356\GOPR0356_frame_000345_depth.png"


def draw_dotted_circle(image, center, radius, color, thickness=2, dot_gap=10):
    # Calculate the number of dots based on the circumference and the gap between dots
    circumference = 2 * np.pi * radius
    num_dots = int(circumference / dot_gap)
    
    for i in range(num_dots):
        # Angle in radians for each dot
        angle = 2 * np.pi * i / num_dots
        
        # Calculate the x and y coordinates for each point on the circle
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        
        # Draw a small circle (dot) at each calculated point
        cv2.circle(image, (x, y), thickness, color, -1)

def highlight_person(image, depth, seg_label, color):
    # Convert segmentation label to RGB if it's not already
    if len(seg_label.shape) == 2 or seg_label.shape[2] == 1:
        seg_label = cv2.cvtColor(seg_label, cv2.COLOR_GRAY2RGB)

    # Define the range for red shades that may represent a person
    lower_red = np.array([150, 0, 0])  # Lower bound of red
    upper_red = np.array([255, 100, 100])  # Upper bound of red

    # Create a mask for the red shades representing people
    mask = cv2.inRange(seg_label, lower_red, upper_red)
    
    # Find contours of the person
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw circles around the person in all images
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = 90 #int(radius * 1.5)

             # Draw dotted circles
            draw_dotted_circle(image, center, radius, (0, 255, 0), thickness=4, dot_gap=20)
            draw_dotted_circle(depth, center, radius, (0, 255, 0), thickness=4, dot_gap=20)
            draw_dotted_circle(seg_label, center, radius, (0, 255, 0), thickness=4, dot_gap=20)
    
    return image, depth, seg_label

def main():
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    # increase font size
    plt.rcParams.update({'font.size': 14})
    
    # Load images
    city_img = cv2.cvtColor(cv2.imread(city_img_filename), cv2.COLOR_BGR2RGB)
    city_depth = cv2.imread(city_depth_filename, cv2.IMREAD_GRAYSCALE)
    city_seg_label = cv2.cvtColor(cv2.imread(city_seg_label_filename), cv2.COLOR_BGR2RGB)
    
    nd_img = cv2.cvtColor(cv2.imread(nd_img_filename), cv2.COLOR_BGR2RGB)
    nd_depth = cv2.imread(nd_depth_filename, cv2.IMREAD_GRAYSCALE)
    nd_seg_label = cv2.cvtColor(cv2.imread(nd_seg_label_filename), cv2.COLOR_BGR2RGB)
    
    dz_img = cv2.cvtColor(cv2.imread(dz_img_filename), cv2.COLOR_BGR2RGB)
    dz_depth = cv2.imread(dz_depth_filename, cv2.IMREAD_GRAYSCALE)
    dz_seg_label = cv2.cvtColor(cv2.imread(dz_seg_label_filename), cv2.COLOR_BGR2RGB)
    
    # Highlight person in each set of images
    city_img, city_depth, city_seg_label = highlight_person(city_img, city_depth, city_seg_label, (255, 0, 0))
    nd_img, nd_depth, nd_seg_label = highlight_person(nd_img, nd_depth, nd_seg_label, (255, 0, 0))
    dz_img, dz_depth, dz_seg_label = highlight_person(dz_img, dz_depth, dz_seg_label, (255, 0, 0))

    # Scale image from cityscapes to match the other images
    city_img = cv2.resize(city_img, (dz_img.shape[1], dz_img.shape[0]))
    city_depth = cv2.resize(city_depth, (dz_img.shape[1], dz_img.shape[0]))
    city_seg_label = cv2.resize(city_seg_label, (dz_img.shape[1], dz_img.shape[0]))
    
    # Display images
    axs[0, 0].imshow(city_img)
    axs[1, 0].imshow(city_depth)
    axs[2, 0].imshow(city_seg_label)

    axs[0, 1].imshow(nd_img)
    axs[1, 1].imshow(nd_depth)
    axs[2, 1].imshow(nd_seg_label)

    axs[0, 2].imshow(dz_img)
    axs[1, 2].imshow(dz_depth)
    axs[2, 2].imshow(dz_seg_label)

    row_labels = ["RGB", "Depth", "Segmentation Label"]
    column_labels = ["Cityscapes", "Nighttime Driving", "Dark Zurich"]
    
    for ax, col in zip(axs[0], column_labels):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], row_labels):
        plt.text(-0.1, 0.5, row, rotation=90, size='large', ha='center', va='center', transform=ax.transAxes)
    
    # Remove axis numbers and ticks
    for ax in axs.flat:
        ax.axis('off')

    # plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0, wspace=0.05)
    plt.show()

if __name__ == "__main__":
    main()