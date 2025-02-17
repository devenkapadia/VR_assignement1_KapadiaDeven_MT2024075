import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_coins(image_path):
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 800))
    image_copy = img.copy()
    
    # Apply Gaussian Blur
    img_blur = cv2.GaussianBlur(img, (11, 11), 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    thresh_inv = cv2.bitwise_not(thresh)  # Inverted binary image
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = {i: cv2.contourArea(cnt) for i, cnt in enumerate(contours)}
    srt = sorted(area.items(), key=lambda x: x[1], reverse=True)
    results = np.array(srt).astype("int")
    num = np.argwhere(results[:, 1] > 50).shape[0]
    
    # Draw contours
    overlay = image_copy.copy()
    for i in range(1, num):
        overlay = cv2.drawContours(overlay, contours, results[i, 0], (0, 255, 0), 3)
    
    print("Number of coins detected:", num - 1)
    
    # Sure foreground extraction
    sure_fg = cv2.erode(thresh, None, iterations=2)
    
    # Marker image (for watershed algorithm if needed)
    markers = cv2.connectedComponents(sure_fg)[1]
    
    # Plot all stages
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 2)
    plt.title("Visualized edges")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 3)
    plt.title("Inverted Binary")
    plt.imshow(thresh_inv, cmap="gray")
    
    plt.subplot(2, 3, 4)
    plt.title("Detected Contours")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 3, 5)
    plt.title("Sure Foreground")
    plt.imshow(sure_fg, cmap="gray")
    
    plt.subplot(2, 3, 6)
    plt.title("Marker Image")
    plt.imshow(markers, cmap="jet")
    
    plt.show()
    
    return num - 1  # Return the number of detected coins


# Example usage
num_coins = detect_coins('coins/coins.jpeg')