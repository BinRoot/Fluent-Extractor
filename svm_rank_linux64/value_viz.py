import sklearn.manifold
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import subprocess, os
import sklearn.cluster

def load_dataset(filename):
    """
    :param filename: *.dat file
    :return: numpy matrix samples x features, and string list of meta info
    """
    with open(filename) as f:
        lines = f.readlines()

    dataset = []
    meta_info = []

    for line in lines:
        line_arr = line.strip().split(' ')
        img_idx = line_arr[0]
        vid_idx = line_arr[1].split(':')[1]
        meta_info.append('{}:{}'.format(vid_idx, img_idx))
        feature_vector = []
        for dim_str in line_arr[2:]:
            feature_vector.append(float(dim_str.split(':')[1]))
        dataset.append(feature_vector)
    dataset = np.asarray(dataset)

    return dataset, meta_info

def compute_values(train_filename):
    prediction_file = 'value_predictions'
    infer_cmd = "./svm_rank_classify {} value_model {}".format(train_filename, prediction_file)
    process = subprocess.Popen(infer_cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

    with open(os.path.join(prediction_file), 'r') as f:
        val_lines = f.readlines()

    print(val_lines)
    vals = map(float, val_lines)
    return vals


if __name__ == '__main__':
    mds = sklearn.manifold.MDS(n_components=2)

    dataset, meta_info = load_dataset('value_train.dat')
    coordinates = mds.fit_transform(dataset)
    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = compute_values('value_train.dat')

    # compute k-means clusters
    kmeans = sklearn.cluster.KMeans(n_clusters=12)
    preds = kmeans.fit_predict(dataset)
    centroids = kmeans.cluster_centers_

    centroids_x = []
    centroids_y = []
    centroids_z = []
    dataset_idxs = []
    for centroid in centroids:
        # find closest index in dataset
        dist = (dataset - centroid) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        dataset_idx = np.argmin(dist)
        dataset_idxs.append(str(dataset_idx))
        centroids_x.append(X[dataset_idx])
        centroids_y.append(Y[dataset_idx])
        centroids_z.append(Z[dataset_idx])
    centroids_x = np.asarray(centroids_x)
    centroids_y = np.asarray(centroids_y)
    centroids_z = np.asarray(centroids_z)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(X, Y, Z, c=Z, marker='.', s=40)
    # ax.scatter3D(X, Y, Z, c=preds, marker='x')
    ax.scatter3D(centroids_x, centroids_y, centroids_z, color='black', marker='o', s=100)
    for i in range(len(centroids_x)):
        ax.text(centroids_x[i], centroids_y[i], centroids_z[i], meta_info[i], color='red', zorder=1)
    plt.title('MDS')
    plt.axis('tight')

    fig_polished = plt.figure()
    ax_polished = Axes3D(fig_polished)
    ax_polished.scatter3D(X, Y, Z, c=Z, marker='.', s=40)
    # ax.scatter3D(X, Y, Z, c=preds, marker='x')
    plt.title('Value Landscape')
    plt.axis('tight')
    plt.show()
