import numpy as np
import matplotlib.pyplot as plt
import cv2

loadedData = 0
preprocessedData = 0
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
    _preprocessData()
    print("Data Preprocessed")
    return loadedData

def _preprocessData():
    resultFlatten = loadedData[0]
    maskFlatten = loadedData[1]
    result = loadedData[2]
    mask = loadedData[3]
    totalPixels = loadedData[4]

    unequalPixels = 0
    totalPositives = 0
    totalTruePositives = 0
    totalResultPositives = 0
    totalMaskPositives = 0
    for i in range(0, totalPixels):
        #resultRounded = round(resultFlatten[i]/255)
        if resultFlatten[i] > 128:
            resultRounded = 1
        else: resultRounded = 0

        if maskFlatten[i] != resultRounded:
            unequalPixels += 1
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        if (maskFlatten[i] or resultRounded): #True if one of the value is not 0, the specific value doesn't matter anymore
            totalPositives += 1
            if (maskFlatten[i] and resultRounded):
                totalTruePositives += 1
                totalResultPositives += 1
                totalMaskPositives += 1
            elif resultRounded:
                totalResultPositives += 1
            elif maskFlatten[i]:
                totalMaskPositives += 1
    print("Total Positives: {} Total True Positives: {}".format(totalPositives, totalTruePositives))
    print("Total Result Positives: {}".format(totalResultPositives))
    print("Total Mask Positives: {}".format(totalMaskPositives))
    global preprocessedData
    preprocessedData = [unequalPixels, totalPositives, totalTruePositives, totalResultPositives, totalMaskPositives]

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

    unequalPixels = preprocessedData[0]
    totalPositives = preprocessedData[1]
    totalTruePositives = preprocessedData[2]
    totalResultPositives = preprocessedData[3]
    totalMaskPositives = preprocessedData[4]

    iou = totalTruePositives/totalPositives
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

    unequalPixels = preprocessedData[0]
    totalPositives = preprocessedData[1]
    totalTruePositives = preprocessedData[2]
    totalResultPositives = preprocessedData[3]
    totalMaskPositives = preprocessedData[4]

    dice = (2 * totalTruePositives)/(totalPositives+totalTruePositives)
    print("The dice for {} is {}".format(result, dice))


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

    unequalPixels = preprocessedData[0]
    totalPositives = preprocessedData[1]
    totalTruePositives = preprocessedData[2]
    totalResultPositives = preprocessedData[3]
    totalMaskPositives = preprocessedData[4]

    accuracy = 1 - (unequalPixels/totalPixels)
    print("The accuracy for {} is {}".format(result, accuracy))

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

    unequalPixels = preprocessedData[0]
    totalPositives = preprocessedData[1]
    totalTruePositives = preprocessedData[2]
    totalResultPositives = preprocessedData[3]
    totalMaskPositives = preprocessedData[4]

    precision = totalTruePositives/totalResultPositives
    print("The precision for {} is {}".format(result, precision))

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

    unequalPixels = preprocessedData[0]
    totalPositives = preprocessedData[1]
    totalTruePositives = preprocessedData[2]
    totalResultPositives = preprocessedData[3]
    totalMaskPositives = preprocessedData[4]

    specificity = (totalPixels - totalPositives)/(totalPixels - totalMaskPositives)
    #print("Total Positives: {} Total True Positives: {}".format(totalPositives, totalMaskPositives))
    print("The specificity for {} is {}".format(result, specificity))

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

    unequalPixels = preprocessedData[0]
    totalPositives = preprocessedData[1]
    totalTruePositives = preprocessedData[2]
    totalResultPositives = preprocessedData[3]
    totalMaskPositives = preprocessedData[4]

    tpr = totalTruePositives/totalMaskPositives
    print("The TPR for {} is {}".format(result, tpr))

    return [mask, tpr]
