import numpy as np
from argparse import ArgumentParser
import sys

from DataProjection import *
from BuildAdjacency import *
from OutlierDetection import *
from BestMap import *
from SpectralClustering import *
from SparseCoefRecovery import *


def SSC(data, nb_clusters, ground_truth=None):

    # The method expects DxN matrix, where columns represent points
    X = data.transpose()
    D, N = X.shape

    r = 0  # Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
    Cst = 1  # Enter 1 to use the additional affine constraint sum(c) == 1
    OptM = 'L1Noise'  # OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
    lmbda = 0.02  # Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
    # Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients


    Xp = DataProjection(X, r, 'NormalProj') # Projects the DxN data matrix into an r-dimensional space
    CMat = SparseCoefRecovery(Xp, Cst, OptM, lmbda) # Sparse optimization
    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    s = ground_truth if ground_truth is not None else np.zeros_like(X[:1]) # ground_truth or vector to ignore afterwards
    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if Fail:
        raise RuntimeError("Something failed")
    
    CKSym = BuildAdjacency(CMatC, 0) # 0 for use all coefficients in adjacency graph, else the number of strongest connections to keep in the graph (something like max of dimensions of the subspaces if it is known, or 1+max ... if affine clusters)
    Grps = SpectralClustering(CKSym, nb_clusters) # shape (N,) containing label of each point

    if ground_truth is not None:
        Grps = BestMap(sc, Grps)
        Missrate = float(np.sum(sc != Grps)) / sc.size
        print("Misclassification rate: {:.4f} %".format(Missrate * 100))

    return Grps    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", help="Path to data file (text file)")
    parser.add_argument("nb_clusters", type=int, help="Number of clusters")
    parser.add_argument("--labels", required=False, default=None, help="Path to ground truth labels (text file)")
    parser.add_argument("--out", required=False, default=None, help="Path where the output labels will be saved (text file)")
    
    args = parser.parse_args(sys.argv[1:])

    data = np.loadtxt(args.data)
    labels = np.loadtxt(args.labels) if args.labels is not None else None

    predicted_labels = SSC(data, nb_clusters=4, ground_truth=labels)
    
    if args.out is not None:
        np.savetxt(args.out, np.expand_dims(predicted_labels, axis=1).astype(int), fmt="%i")
    else:
        print(predicted_labels.astype(int))