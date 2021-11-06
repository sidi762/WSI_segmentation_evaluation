import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import os

#result file name is the corresponding mask file name with a probmap_0.3_ in the front

def calculatePixelAccuracy(result, mask):
    unequalPixels = 0
    maskFlatten = np.load(mask).flatten()
    print(mask + " loaded")
    resultFlatten = np.load(result).flatten()
    print(result + " loaded")
    totalPixels = resultFlatten.shape[0]
    for i in range(0, totalPixels):
        if maskFlatten[i] == 255:
            maskFlatten[i] = 1 #Convert all the 255s to 1
        if maskFlatten[i] != resultFlatten[i]:
            unequalPixels += 1

    accuracy = 1 - (unequalPixels/totalPixels)
    print("The accuracy for {} is {}".format(mask, accuracy))
    return [mask,accuracy]


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    resultPath = "./results"
    maskPath = "./masks"
    results = []
    for root, dirs, files in os.walk(maskPath):
        for file in files:
            resultFile = 'probmap_0.3_' + file
            maskFilePath = os.path.join(root, file)
            resultFilePath = os.path.join(resultPath,resultFile)
            results.append(pool.apply_async(calculatePixelAccuracy, (resultFilePath, maskFilePath)))
    pool.close()
    pool.join()
    resultOutput = open('pixel_accuracy.txt', mode='w+',encoding='utf-8')
    total = 0
    numOfResults = 0
    for res in results:
        resultOutput.write("{}: {}\n".format(res.get()[0], res.get()[1]))
        total += res.get()[1]
        numOfResults += 1
    resultOutput.write("Mean: {}\n".format(total/numOfResults))
    resultOutput.close()
