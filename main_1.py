import os
import cv2
import numpy as np

def compute_fingerprint_score(img1, img2, sift):
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    if descriptors_1 is None or descriptors_2 is None:
        return 0, [], [], []

    # Normalize descriptors
    descriptors_1 = cv2.normalize(descriptors_1, None, norm_type=cv2.NORM_L2)
    descriptors_2 = cv2.normalize(descriptors_2, None, norm_type=cv2.NORM_L2)

    # FLANN matcher with adjusted parameters
    index_params = {'algorithm': 1, 'trees': 5}
    search_params = {'checks': 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Use RANSAC to filter matches
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        ransac_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
    else:
        ransac_matches = good_matches

    keypoints_count = min(len(keypoints_1), len(keypoints_2))
    score = (len(ransac_matches) / keypoints_count * 100) if keypoints_count > 0 else 0

    return score, keypoints_1, keypoints_2, ransac_matches


# Load the sample fingerprint image
sample = cv2.imread("SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP")
sift = cv2.SIFT_create()

# Metrics
genuine_scores = []
altered_scores = []

genuine_attempts, correct_genuine_matches, false_rejections = 0, 0, 0
altered_attempts, false_acceptances = 0, 0

# Genuine fingerprints evaluation
for file in os.listdir("SOCOFing/Real")[:1000]:
    genuine_attempts += 1
    fingerprint_image = cv2.imread(os.path.join("SOCOFing/Real", file))
    if fingerprint_image is None:
        continue

    score, kp1, kp2, matches = compute_fingerprint_score(sample, fingerprint_image, sift)
    genuine_scores.append(score)

genuine_threshold = np.mean(genuine_scores) - np.std(genuine_scores)

# Altered fingerprints evaluation
for folder in ["SOCOFing/Altered/Altered-Easy", "SOCOFing/Altered/Altered-Medium", "SOCOFing/Altered/Altered-Hard"]:
    for file in os.listdir(folder)[:1000]:
        altered_attempts += 1
        fingerprint_image = cv2.imread(os.path.join(folder, file))
        if fingerprint_image is None:
            continue

        score, kp1, kp2, matches = compute_fingerprint_score(sample, fingerprint_image, sift)
        altered_scores.append(score)

altered_threshold = np.mean(altered_scores) + np.std(altered_scores)

# Performance calculations
false_rejections = sum(1 for score in genuine_scores if score < genuine_threshold)
false_acceptances = sum(1 for score in altered_scores if score > altered_threshold)

frr = (false_rejections / genuine_attempts) * 100 if genuine_attempts > 0 else 0
far = (false_acceptances / altered_attempts) * 100 if altered_attempts > 0 else 0
accuracy = ((genuine_attempts - false_rejections + altered_attempts - false_acceptances) / (
            genuine_attempts + altered_attempts)) * 100

eer = (frr + far) / 2  # Approximate Equal Error Rate

print("False Rejection Rate (FRR): {:.2f}%".format(frr))
print("False Acceptance Rate (FAR): {:.2f}%".format(far))
print("Equal Error Rate (EER): {:.2f}%".format(eer))
print("Accuracy: {:.2f}%".format(accuracy))

# Display result if matches are found
if matches:
    result = cv2.drawMatches(sample, kp1, fingerprint_image, kp2, matches, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matches found.")
