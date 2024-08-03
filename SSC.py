import numpy as np
from argparse import ArgumentParser
import sys

from DataProjection import *
from BuildAdjacency import *
from OutlierDetection import *
from BestMap import *
from SpectralClustering import *
from SparseCoefRecovery import *


def SSC(data, nb_clusters, ground_truth=None, r=0, Proj="NormalProj", Cst=1, OptM="Lasso", lmbda=0.01, K=0):
    """
    Main function for SSC execution.
    
    Arguments:
    data : (N, D) numpy.ndarray containing the datapoints
    nb_clusters : the number of clusters
    ground_truth : (N,) numpy.ndarray containing ground-truth labels
    r : The projection dimension if data is projected before being clustered, 0 is not projection.
    Proj : can be {'PCA', 'NormalProj', 'BernoulliProj'}
    Cst : 1 to use additional affine contraint sum(c) == 1, 0 otherwise
    OptM : can be {'L1Perfect','L1Noise','Lasso','L1ED'}
    lmbda : Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
    K : 0 for use all coefficients in adjacency graph, else the number of strongest connections to keep in the graph (something like max of dimensions of the subspaces if it is known, or 1+max ... if affine clusters)
    """

    # The method expects DxN matrix, where columns represent points
    X = data.transpose()
    D, N = X.shape

    Xp = DataProjection(X, r, Proj) # Projects the DxN data matrix into an r-dimensional space
    CMat = SparseCoefRecovery(Xp, Cst, OptM, lmbda) # Sparse optimization
    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    s = ground_truth if ground_truth is not None else np.zeros_like(X[:1]) # ground_truth or vector to ignore afterwards
    CMatC, sc, OutlierIndx, Fail = OutlierDetection(CMat, s)

    if Fail:
        raise RuntimeError("Something failed")
    
    CKSym = BuildAdjacency(CMatC, K) 
    Grps = SpectralClustering(CKSym, nb_clusters) # shape (N,) containing label of each point

    if ground_truth is not None:
        Grps = BestMap(sc, Grps)
        Missrate = float(np.sum(sc != Grps)) / sc.size
        print("Misclassification rate: {:.4f} %".format(Missrate * 100))

    return Grps    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data file (text file)")
    parser.add_argument("--labels", required=False, default=None, help="Path to ground truth labels (text file)")
    parser.add_argument("--nb_clusters", required=True, type=int, help="Number of clusters")
    parser.add_argument("--out", required=False, default=None, help="Path where the output labels will be saved (text file)")
    
    args = parser.parse_args(sys.argv[1:])

    data = np.loadtxt(args.data)
    labels = np.loadtxt(args.labels) if args.labels is not None else None

    predicted_labels = SSC(data, nb_clusters=4, ground_truth=labels)
    
    if args.out is not None:
        np.savetxt(args.out, np.expand_dims(predicted_labels, axis=1).astype(int), fmt="%i")
    else:
        print(predicted_labels.astype(int))