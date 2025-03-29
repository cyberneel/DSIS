# import dependencies
import easyocr
import cv2
import os
import numpy as np

current_dir = os.path.dirname(__file__)


print(cv2.getBuildInformation())

# Check if OpenCV is built with CUDA support
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Get the CUDA device
    device = cv2.cuda.getDevice()
    # Retrieve the device name
    device_name = cv2.cuda.printCudaDeviceInfo(device)
    print(f"CUDA Device Name: {device_name}")
else:
    print("No CUDA-enabled GPU found.")

def preprocess(img):
    # First filter only red color
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])
    mask = cv2.inRange(img, lower_red, upper_red)
    img = cv2.bitwise_and(img, img, mask=mask)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])

img = cv2.imread(os.path.join(current_dir, "./Design/Design.jpg"))

contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    sides = len(hull)

    if 6 > sides > 3 and sides + 2 == len(approx):
        arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
        if arrow_tip:
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
            cv2.circle(img, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

cv2.imshow("arrows.jpeg", preprocess(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
