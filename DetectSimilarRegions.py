import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Global variables to store the cropping state and the cropped region
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropped_region = None

def select_region(event, x, y, flags, param):
    """A callback function for mouse events.

        Args:
            event: The event type, such as cv2.EVENT_LBUTTONDOWN or cv2.EVENT_MOUSEMOVE.
            x: The x-coordinate of the mouse pointer.
            y: The y-coordinate of the mouse pointer.
            flags: Event-specific flags.
            param: User-defined parameter.
        """

    global x_start, y_start, x_end, y_end, cropping, cropped_region, img

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        cropped_region = clone[y_start:y_end, x_start:x_end]
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("Select Region", img)

def calculate_mean_tamura_features(region):
    """Calculates the mean Tamura features for a given image region.

        Args:
            region: A NumPy array representing the image region.

        Returns:
            A NumPy array containing the mean Tamura features for the given image region.
        """

    if region is None or region.size == 0:
        return np.zeros(6)  # Return zeros if the region is invalid

    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_region, [1], [0], symmetric=True, normed=True)

    # Calculate the Tamura features from the GLCM.
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    linelikeness = 1.0 - homogeneity
    regularity = 1.0 - (1.0 / (1.0 + contrast))
    roughness = np.sqrt(contrast)
    return np.array([contrast, homogeneity, dissimilarity, linelikeness, regularity, roughness])

def find_similar_regions_mean(image, selected_region, mean_tamura_features_selected, threshold=0.2):
    """Finds all regions in the given image that have similar mean Tamura features to the selected region.

        Args:
            image: A NumPy array representing the image.
            selected_region: A NumPy array representing the selected region.
            mean_tamura_features_selected: A NumPy array containing the mean Tamura features of the selected region.
            threshold: A similarity threshold.

        Returns:
            A list of (x, y) coordinates of the regions in the image that have similar mean Tamura features to the selected region.
        """

    h, w = image.shape[:2]
    similar_regions = []

    if selected_region is None:
        return similar_regions

    # Iterate over all pixels in the image and compare the mean Tamura features of the current pixel's region
    # to the mean Tamura features of the selected region. If the similarity score is less than the threshold,
    # add the current pixel's coordinates to the list of similar regions.
    for y in range(0, h - selected_region.shape[0]):
        for x in range(0, w - selected_region.shape[1]):
            region = image[y:y + selected_region.shape[0], x:x + selected_region.shape[1]]
            mean_tamura_features = calculate_mean_tamura_features(region)

            similarity_score = np.linalg.norm(mean_tamura_features_selected - mean_tamura_features)

            if similarity_score < threshold:
                similar_regions.append((x, y))

    return similar_regions

image_path = "C:\\Users\\emy7u\\Downloads\\b.jpg"
img = cv2.imread(image_path)
clone = img.copy()

cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", select_region)
cv2.imshow("Select Region", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate mean Tamura features for the selected region
mean_tamura_features_selected = calculate_mean_tamura_features(cropped_region)

# Find regions with similar mean textures
threshold = 0.2
similar_regions_mean = find_similar_regions_mean(img, cropped_region, mean_tamura_features_selected, threshold)

# Create a binary mask
binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)

for x, y in similar_regions_mean:
    binary_mask[y:y + cropped_region.shape[0], x:x + cropped_region.shape[1]] = 255

# Display the binary mask
cv2.imshow("Binary Mask", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
