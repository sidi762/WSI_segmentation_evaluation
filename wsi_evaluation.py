import numpy as np
import matplotlib.pyplot as plt
import cv2

loadedData = 0
dataLoaded = 0

def loadData(result, mask):
    loadedMask = np.load(mask)
    maskShape = loadedMask.shape
    print(mask + " loaded")
    print("Mask Shape: {}".format(maskShape))

    loadedResult = np.load(result)
    resultShape = loadedResult.shape
    print(result + " loaded")
    print("Result Shape: {}".format(resultShape))

    if maskShape != resultShape:
        loadedMask = cv2.resize(loadedMask, (resultShape[1], resultShape[0]))
        print("Input shape different from mask shape, mask resized to {}.".format(loadedMask.shape))

    maskFlatten = loadedMask.flatten()
    print("Total Pixels in Mask (after resizing, if resized): {}".format(maskFlatten.shape[0]))

    resultFlatten = loadedResult.flatten()
    totalPixels = resultFlatten.shape[0]
    print("Total Pixels in Result: {}".format(totalPixels))
    global loadedData
    loadedData = [resultFlatten, maskFlatten, result, mask, totalPixels]
    global dataLoaded
    dataLoaded = 1
    return loadedData


def calculateIoU():
    #for 2D image only
    if dataLoaded == 0:
        print("Error: Data not loaded! Run wsi_evaluation.loadData() first.")
        return
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    unequalPixels = 0
    totalPositives = 0
    totalTruePositives = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        resultRounded = round(resultFlatten[i]/255)
        if (maskFlatten[i] or resultRounded): #True if one of the value is not 0, the specific value doesn't matter anymore
            totalPositives += 1
            if (maskFlatten[i] and resultRounded):
                totalTruePositives += 1

    iou = totalTruePositives/totalPositives
    #print("Total Positives: {} Total True Positives: {}".format(totalPositives, totalTruePositives))
    print("The iou for {} is {}".format(result, iou))
    return [mask,iou]

def calculateDICE():
    #for 2D image only
    if dataLoaded == 0:
        print("Error: Data not loaded! Run wsi_evaluation.loadData() first.")
        return
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    totalPositives = 0
    totalTruePositives = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        resultRounded = round(resultFlatten[i]/255)
        if (maskFlatten[i] or resultRounded): #True if one of the value is not 0, the specific value doesn't matter anymore
            totalPositives += 1
            if (maskFlatten[i] and resultRounded):
                totalTruePositives += 1

    dice = (2 * totalTruePositives)/(totalPositives+totalTruePositives)
    #print("Total Positives: {} Total True Positives: {}".format(totalPositives, totalTruePositives))
    print("The dice for {} is {}".format(mask, dice))
    return [mask, dice]

def calculatePixelAccuracy():
    #for 2D image only
    if dataLoaded == 0:
        print("Error: Data not loaded! Run wsi_evaluation.loadData() first.")
        return
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    unequalPixels = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        resultRounded = round(resultFlatten[i]/255)
        if maskFlatten[i] != resultRounded:
            unequalPixels += 1

    accuracy = 1 - (unequalPixels/totalPixels)
    print("The accuracy for {} is {}".format(mask, accuracy))
    return [mask,accuracy]

def calculatePrecision():
    #for 2D image only
    if dataLoaded == 0:
        print("Error: Data not loaded! Run wsi_evaluation.loadData() first.")
        return
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    unequalPixels = 0
    totalResultPositives = 0
    totalTruePositives = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        if (round(resultFlatten[i]/255)): #True if one of the value is not 0, the specific value doesn't matter anymore
            totalResultPositives += 1
            if (maskFlatten[i]):
                totalTruePositives += 1

    precision = totalTruePositives/totalResultPositives
    #print("Total Result Positives: {} Total True Positives: {}".format(totalResultPositives, totalTruePositives))
    print("The precision for {} is {}".format(mask, precision))
    return [mask, precision]

def calculateSpecificity():
    #for 2D image only
    if dataLoaded == 0:
        print("Error: Data not loaded! Run wsi_evaluation.loadData() first.")
        return
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    unequalPixels = 0
    totalMaskPositives = 0
    totalPositives = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        if (maskFlatten[i]):
            totalPositives += 1
            totalMaskPositives += 1
        elif (round(resultFlatten[i]/255)):
            totalPositives += 1


    specificity = (totalPixels - totalPositives)/(totalPixels - totalMaskPositives)
    #print("Total Positives: {} Total True Positives: {}".format(totalPositives, totalMaskPositives))
    print("The specificity for {} is {}".format(mask, specificity))
    return [mask, specificity]

def calculateTPR():
    #for 2D image only
    if dataLoaded == 0:
        print("Error: Data not loaded! Run wsi_evaluation.loadData() first.")
        return
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    unequalPixels = 0
    totalMaskPositives = 0
    totalTruePositives = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        if (maskFlatten[i]): #True if one of the value is not 0, the specific value doesn't matter anymore
            totalMaskPositives += 1
            if (round(resultFlatten[i]/255)):
                totalTruePositives += 1

    tpr = totalTruePositives/totalMaskPositives
    #print("Total Mask Positives: {} Total True Positives: {}".format(totalMaskPositives, totalTruePositives))
    print("The TPR for {} is {}".format(mask, tpr))
    return [mask, tpr]
