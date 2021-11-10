import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import os

#result file name is the corresponding mask file name with a probmap_0.3_ in the front

def calculateIoU(result, mask):
    unequalPixels = 0
    maskFlatten = np.load(mask).flatten()
    print(mask + " loaded")
    #maskFlatten = np.array([0,0,0,1,1,1])
    resultFlatten = np.load(result).flatten()
    #resultFlatten = np.array([0,0,1,1,1,0]) #For testing
    print(result + " loaded")
    totalPixels = resultFlatten.shape[0]
    print("Total Pixels: {}".format(totalPixels))
    totalPositives = 0
    totalTruePositives = 0
    for i in range(0, totalPixels):
        #if maskFlatten[i] == 255:
        #    maskFlatten[i] = 1 #Convert all the 255s to 1
        if (maskFlatten[i] or resultFlatten[i]): #True if one of the value is not 0, the specific value doesn't matter anymore
            totalPositives += 1
            if (maskFlatten[i] and resultFlatten[i]):
                totalTruePositives += 1

    iou = totalTruePositives/totalPositives
    print("Total Positives: {} Total True Positives: {}".format(totalPositives, totalTruePositives))
    print("The iou for {} is {}".format(mask, iou))
    return [mask,iou]


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=8)
    #resultPath = "./results"
    resultPath = "./original_post_npy"
    resultPrefix = "original_post_0.3_"
    maskPath = "./masks"
    results = []
    for root, dirs, files in os.walk(maskPath):
        for file in files:
            resultFile = resultPrefix + file
            maskFilePath = os.path.join(root, file)
            resultFilePath = os.path.join(resultPath,resultFile)
            if os.path.exists(resultFilePath):
                results.append(pool.apply_async(calculateIoU, (resultFilePath, maskFilePath)))

    pool.close()
    pool.join()
    resultOutput = open((resultPrefix+'iou.txt'), mode='w+',encoding='utf-8')
    total = 0
    numOfResults = 0
    for res in results:
        resultOutput.write("{}: {}\n".format(res.get()[0], res.get()[1]))
        total += res.get()[1]
        numOfResults += 1
    resultOutput.write("Average: {}\n".format(total/numOfResults))
    resultOutput.close()
