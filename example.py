import wsi_evaluation

if __name__ == '__main__':
    #wsi_evaluation.loadData("./000273-3c-2/prob_000273-3c-2.npy", "./000273-3c-2/000273-3c-2.npy")
    #wsi_evaluation.loadData("./000523-2a-1/prob_crf_000523-2a-1.npy", "./000523-2a-1/000523-2a-1.npy")
    wsi_evaluation.loadData("./000312-1c-2/prob_crf_000312-1c-2.npy", "./000312-1c-2/000312-1c-2.npy")
    wsi_evaluation.calculateIoU()
    wsi_evaluation.calculateDICE()
    wsi_evaluation.calculatePixelAccuracy()
    wsi_evaluation.calculatePrecision()
    wsi_evaluation.calculateSpecificity()
    wsi_evaluation.calculateTPR()
