import cv2
import numpy as np


def hls_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    _, mask_white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    mask_yellow = cv2.inRange(hls, np.array([10, 0, 60]), np.array([30, 255, 255]))
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


def roi_mask(image):
    height = image.shape[0]
    width = image.shape[1]
    left_lane_roi = np.array([
        [50, height - 30],
        [width / 2, height / 2 + 50],
        [width / 2 - 70, height - 30]
    ], np.int32)
    right_lane_roi = np.array([
        [width - 50, height - 30],
        [width / 2, height / 2 + 50],
        [width / 2 + 70, height - 30]
    ], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [left_lane_roi, right_lane_roi], 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def implement_lines(image, lines):
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        return image
    except Exception:
        return image


def line_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters

        y1 = image.shape[0]
        y2 = int(y1 * 0.5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    except Exception:
        return np.array([0, 0, 0, 0])


def line_filter(image, lines):
    left = []
    right = []
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < -0.3:
                left.append((slope, intercept))
            elif 0.3 < slope:
                right.append((slope, intercept))

            left_avg = np.average(left, axis=0)
            right_avg = np.average(right, axis=0)
    except Exception:
        left_avg = (0, 0)
        right_avg = (0, 0)

    left_lane = line_coordinates(image, left_avg)
    right_lane = line_coordinates(image, right_avg)

    return np.array([left_lane, right_lane])


cap = cv2.VideoCapture('project_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        lane_image = np.copy(frame)
        blur_image = cv2.bilateralFilter(lane_image, 5, 100, 100)
        hls_masked = hls_mask(blur_image)
        gray_image = cv2.cvtColor(hls_masked, cv2.COLOR_BGR2GRAY)
        canny_image = cv2.Canny(gray_image, 150, 300)
        roi_image = roi_mask(canny_image)

        lines = cv2.HoughLinesP(roi_image,
                                rho=2,
                                theta=np.pi / 180,
                                threshold=40,
                                lines=np.array([]),
                                minLineLength=3,
                                maxLineGap=5)
        lines_filtered = line_filter(lane_image, lines)
        lines_added = implement_lines(frame, lines)
        lane_detected = implement_lines(lane_image, lines_filtered)

        cv2.imshow('canny', cv2.resize(roi_image, (640, 360)))
        cv2.imshow('lines', cv2.resize(lines_added, (640, 360)))
        cv2.imshow('res', cv2.resize(lane_detected, (640, 360)))

        if cv2.waitKey(5) & 0xFF == 27:
            break

    else:
        break