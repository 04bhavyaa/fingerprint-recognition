import os
import cv2
import numpy as np

# Load the sample fingerprint image
sample = cv2.imread("SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP")

# Initialize variables
best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None
sift = cv2.SIFT_create()

# Performance metrics
genuine_attempts = 0
correct_genuine_matches = 0
false_rejections = 0
altered_attempts = 0
false_acceptances = 0

genuine_threshold = int(input("Enter threshold for genuine matches (default 30): ") or 30)
altered_threshold = int(input("Enter threshold for altered matches (default 20): ") or 20)

for file in os.listdir("SOCOFing/Real")[:1000]:
    genuine_attempts += 1
    fingerprint_image = cv2.imread(os.path.join("SOCOFing/Real", file))
    if fingerprint_image is None:
        continue

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = [p for p, q in matches if p.distance < 0.80 * q.distance]

    keypoints_count = min(len(keypoints_1), len(keypoints_2))
    if keypoints_count > 0:
        score = len(match_points) / keypoints_count * 100
        if score > genuine_threshold:
            correct_genuine_matches += 1
            if score > best_score:
                best_score = score
                filename = file
                image = fingerprint_image
                kp1, kp2, mp = keypoints_1, keypoints_2, match_points
        else:
            false_rejections += 1

for folder in ["SOCOFing/Altered/Altered-Easy", "SOCOFing/Altered/Altered-Medium", "SOCOFing/Altered/Altered-Hard"]:
    for file in os.listdir(folder)[:1000]:
        altered_attempts += 1
        fingerprint_image = cv2.imread(os.path.join(folder, file))
        if fingerprint_image is None:
            continue

        keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
        match_points = [p for p, q in matches if p.distance < 0.80 * q.distance]

        keypoints_count = min(len(keypoints_1), len(keypoints_2))
        if keypoints_count > 0:
            score = len(match_points) / keypoints_count * 100
            if score > altered_threshold:
                false_acceptances += 1

# Performance calculations
total_attempts = genuine_attempts + altered_attempts
correct_rejections = altered_attempts - false_acceptances
correct_recognitions = correct_genuine_matches

frr = (false_rejections / genuine_attempts) * 100 if genuine_attempts > 0 else 0
far = (false_acceptances / altered_attempts) * 100 if altered_attempts > 0 else 0
accuracy = ((correct_recognitions + correct_rejections) / total_attempts) * 100 if total_attempts > 0 else 0

threshold_range = np.linspace(min(genuine_threshold, altered_threshold), max(genuine_threshold, altered_threshold),
                              num=50)
min_diff = float("inf")
eer = 0
for threshold in threshold_range:
    temp_frr = (false_rejections / genuine_attempts) * 100 if genuine_attempts > 0 else 0
    temp_far = (false_acceptances / altered_attempts) * 100 if altered_attempts > 0 else 0
    if abs(temp_frr - temp_far) < min_diff:
        min_diff = abs(temp_frr - temp_far)
        eer = (temp_frr + temp_far) / 2

print("Best Match:", filename)
print("Score:", best_score)
print("False Rejection Rate (FRR): {:.2f}%".format(frr))
print("False Acceptance Rate (FAR): {:.2f}%".format(far))
print("Equal Error Rate (EER): {:.2f}%".format(eer))
print("Accuracy: {:.2f}%".format(accuracy))

if image is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matches found.")
