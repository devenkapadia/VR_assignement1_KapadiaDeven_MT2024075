# Image Processing Assignment using OpenCV  

This repository contains two image processing projects using OpenCV: **Coin Detection** and **Image Stitching**.

---  

## 1. Coin Detection  
Part 1: Use computer vision techniques to Detect, segment, and count coins from an
image containing scattered Indian coins.

### Methods Chosen  
- **Gaussian Blur**: Reduces noise for better edge detection.  
- **Thresholding**: Segments the image into binary form.  
- **Contour Detection**: Identifies object boundaries and filters out noise.  
- **Sure Foreground Extraction**: Uses erosion to refine object detection.  
- **Marker Image Creation**: Creates markers for future advanced segmentation.  

### Results & Observations  
- Successfully detected coins in the image.  
- The use of Gaussian Blur improved segmentation.  
- Some small false detections may occur due to image noise.  

### Visual Outputs  
- Original Image  
- Processed Image with Contours  
- Thresholded Binary Image  
- Sure Foreground Extraction  

### How to Run  
1. Install dependencies:  
   ```bash
   pip install numpy opencv-python matplotlib
   ```  
2. Place the coin image in the project directory.  
3. Run the script:  
   ```python
   num_coins = detect_coins('path/to/image.jpeg')
   print(f'Number of coins detected: {num_coins}')
   ```  
4. The output will show the detected coins and display intermediate steps.  

---  

## 2. Image Stitching  
Part 2: Create a stitched panorama from multiple overlapping images.

### Methods Chosen  
- **Feature Detection (SIFT)**: Detects keypoints and descriptors.  
- **Feature Matching (FLANN-based Matcher)**: Finds good feature correspondences.  
- **Homography Estimation (RANSAC)**: Computes the transformation matrix.  
- **Image Warping & Stitching**: Warps and blends images into a panorama.  
- **Cropping**: Removes black borders from the stitched image.  

### Results & Observations  
- Successfully stitched images into a panorama.  
- The quality of matches impacts the final panorama.  
- Some misalignment may occur if the images lack enough key features.  

### Visual Outputs  
- Feature Matches Between Images  
- Stitched Panorama Image  
- Cropped Final Output  

### How to Run  
1. Install dependencies:  
   ```bash
   pip install numpy opencv-python matplotlib
   ```  
2. Place images in the `pan1/` or `pan2/` folder.  
3. Run the script:  
   ```python
   python script.py
   ```  
4. The final panorama will be saved as `panorama.jpg` and displayed in a matplotlib figure.  

---  

## License  
This project is a part of assignment given in Visual Recognition Subject
