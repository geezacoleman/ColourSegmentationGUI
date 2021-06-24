import numpy as np
import imutils
import cv2

def exG(image, thresholdMin=13, thresholdMax=200, minArea=100, saturation=1.0, brightness=1.0, blur=1, externalOnly=False):
    """
    Uses an exG or excess green index and contour detection to determine green objects in the image. Min and Max
    thresholds are provided.
    :param image: image array
    :param thresholdMin: min exG threshold
    :param thresholdMax: max exG threshold
    :return: returns  lsits of contours, bounding boxes, centres of objects and the modified image
    """
    # using array slicing to split into channels
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)

        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 2] = hsv[:, :, 2] * brightness

        adjusted = np.clip(hsv, [0, 0, 0], [255, 255, 255])
        adjusted = adjusted.astype(np.uint8)
        blurTuple = (blur, blur)
        #blurred = cv2.medianBlur(adjusted, blur, cv2.BORDER_DEFAULT)
        blurred = cv2.GaussianBlur(adjusted, blurTuple, cv2.BORDER_DEFAULT)
        blurred = cv2.bilateralFilter(blurred, blur, 50, 50)
        hsvThresh, _ = hsv_threshold(image=blurred)
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)

        red = blurred[:, :, 0].astype(np.float32)
        green = blurred[:, :, 1].astype(np.float32)
        blue = blurred[:, :, 2].astype(np.float32)
        # calculate the exG index and the convert to an absolute 0 - 255 scale
        exG = 2 * green - red - blue
        exG = np.clip(exG, 0, 255)
        exG = exG.astype(np.uint8)

        # run the thresholds provided
        output = cv2.inRange(exG, thresholdMin, thresholdMax)
    except Exception as e:
        print(e)

    # combine HSV and exg
    thresholdOut = output & hsvThresh
    # find all the contours
    cnts = cv2.findContours(thresholdOut.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
    if externalOnly:
        for c in cnts:
            # filter based on total area of contour
            if cv2.contourArea(c) > minArea:
                cv2.drawContours(thresholdOut, [c], -1, (255), -1)

    combined = cv2.bitwise_and(image, image, mask=thresholdOut)
    display = cv2.bitwise_and(adjusted, adjusted, mask=thresholdOut)
    return cnts, combined, thresholdOut, display


def hsv_threshold(image, hueMin=38, hueMax=75, brightnessMin=40, brightnessMax=255, saturationMin=40, saturationMax=255):
    hue = image[:, :, 0]
    sat = image[:, :, 1]
    val = image[:, :, 2]
    # cv2.imshow('hue', hue)
    # cv2.imshow('sat', sat)
    # cv2.imshow('val', val)

    hueThresh = cv2.inRange(hue, hueMin, hueMax)
    satThresh = cv2.inRange(sat, saturationMin, saturationMax)
    valThresh = cv2.inRange(val, brightnessMin, brightnessMax)

    outThresh = satThresh & valThresh & hueThresh
    #cv2.imshow('HSV Out', outThresh)
    return outThresh, True


def hsv_segmentation(image, hmin=0, hmax=128, smin=0, smax=255, vmin=50, vmax=200, minArea=1, saturation=1.0, brightness=1.0, blur=1, externalOnly=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype('float32')
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 2] = hsv[:, :, 2] * brightness

    adjusted = np.clip(hsv, [0, 0, 0], [255, 255, 255])
    adjusted = adjusted.astype('uint8')

    adjusted = cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)
    blurred = cv2.medianBlur(adjusted, blur)

    lowerThresh = np.array([hmin, smin, vmin])
    upperThresh = np.array([hmax, smax, vmax])

    thresholdOut = cv2.inRange(blurred, lowerThresh, upperThresh)

    cnts = cv2.findContours(thresholdOut.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
    if externalOnly:
        for c in cnts:
            # filter based on total area of contour
            if cv2.contourArea(c) > minArea:
                cv2.drawContours(thresholdOut, [c], -1, (255), -1)

    combined = cv2.bitwise_and(image, image, mask=thresholdOut)
    display = cv2.bitwise_and(adjusted, adjusted, mask=thresholdOut)
    return cnts, combined, thresholdOut, display

def crop_to_contour(mask, maskedImage, contour):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)

    blankMask = np.zeros(mask.shape[:2], dtype="uint8")
    cv2.drawContours(blankMask, [contour], -1, 255, -1)
    startX, startY, boxW, boxH = cv2.boundingRect(contour)
    weedOnly = cv2.bitwise_and(maskedImage, maskedImage, mask=blankMask)

    return weedOnly, blankMask, startX, startY, boxW, boxH