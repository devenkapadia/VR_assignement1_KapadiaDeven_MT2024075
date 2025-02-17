import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_match_features(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def draw_matches(img1, img2):
    keypoints1, keypoints2, good_matches = detect_and_match_features(img1, img2)

    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img

def stitch_images(img1, img2):
    keypoints1, keypoints2, good_matches = detect_and_match_features(img1, img2)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        height, width, _ = img2.shape
        result = cv2.warpPerspective(img1, H, (width * 2, height))

        result[0:height, 0:width] = img2

        # Convert to grayscale and create mask to find non-black pixels
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours and bounding box of the stitched region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Crop the black regions
        result_cropped = result[y:y+h, x:x+w]

        return result_cropped
    else:
        print("Not enough matches found!")
        return None


# img1 = cv2.imread("pan2/i1.png")
# img2 = cv2.imread("pan2/i2.png")
# img3 = cv2.imread("pan2/i3.png")

img1 = cv2.imread("pan1/i1.jpeg")
img2 = cv2.imread("pan1/i2.jpeg")
img3 = cv2.imread("pan1/i3.jpeg")

# Draw feature matches
matches_img1 = draw_matches(img1, img2)
matches_img2 = draw_matches(img2, img3)

# Stitch images
panorama1 = stitch_images(img2, img1)
panorama2 = stitch_images(img3, img2)

if panorama1 is not None and panorama2 is not None:
    
    panorama_final = stitch_images(img3, panorama1)

    if panorama_final is not None:
        cv2.imwrite("panorama.jpg", panorama_final)

        plt.figure(figsize=(30,10))

        plt.subplot(2,3,1)
        plt.title("Matches between Img1 & Img2")
        plt.imshow(cv2.cvtColor(matches_img1,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,2)
        plt.title("Matches between Img2 & Img3")
        plt.imshow(cv2.cvtColor(matches_img2,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,3)
        plt.title("PANAROMA")
        plt.imshow(cv2.cvtColor(panorama_final,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,4)
        plt.title("PANAROMA1")
        plt.imshow(cv2.cvtColor(panorama1,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,5)
        plt.title("PANAROMA2")
        plt.imshow(cv2.cvtColor(panorama2,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,6)
        plt.axis("off")  # Empty plot to maintain layout

        plt.show()
    else:
        print("Image stitching failed for third image!")
else:
    print("Image stitching failed for first two images!")
