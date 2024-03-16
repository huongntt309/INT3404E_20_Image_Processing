import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image from BGR to RGB (OpenCV reads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Remember to use plt.show() to display the image
    """

    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        # Display the image using imshow
        plt.imshow(image)
        
    # Set the title of the plot
    plt.title(title)
    # Hide the axes ticks
    plt.xticks([])
    plt.yticks([])
    # Show the plot
    plt.show()


def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    
    # Extract R, G, B channels
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Calculate the grayscale value using the provided equation
    img_gray = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert the dtype to uint8
    img_gray = img_gray.astype(np.uint8)
    return img_gray


def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    # Convert the image back to BGR format 
    if len(image.shape) == 2:
        # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    else:
        # If RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite(output_path, image)


def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    
    return flipped_image


def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    # Get the height and width of the image
    height, width = image.shape[:2]
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    
    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
