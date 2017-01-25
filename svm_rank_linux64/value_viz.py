import sklearn.manifold
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import subprocess, os
import sklearn.cluster
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib._png import read_png

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def load_dataset(filename):
    """
    :param filename: *.dat file
    :return: numpy matrix samples x features, and string list of meta info
    """
    with open(filename) as f:
        lines = f.readlines()

    dataset = []
    meta_info = []
    all_indices = []
    # [[0, 1, 2], [3, 4], ...]
    # get corresponding mds points

    indices = []
    prev_vid_idx = None
    for line_idx, line in enumerate(lines):
        line_arr = line.strip().split(' ')
        img_idx = line_arr[0]
        vid_idx = line_arr[1].split(':')[1]
        if prev_vid_idx is None:
            indices = []
        elif prev_vid_idx != vid_idx:
            all_indices.append(indices)
            indices = []
        indices.append(line_idx)

        meta_info.append('{}:{}'.format(vid_idx, img_idx))
        feature_vector = []
        for dim_str in line_arr[2:]:
            feature_vector.append(float(dim_str.split(':')[1]))
        dataset.append(feature_vector)
        prev_vid_idx = vid_idx
    dataset = np.asarray(dataset)

    return dataset, meta_info, all_indices


def compute_values(train_filename):
    prediction_file = 'value_predictions'
    infer_cmd = "./svm_rank_classify {} value_model {}".format(train_filename, prediction_file)
    process = subprocess.Popen(infer_cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

    with open(os.path.join(prediction_file), 'r') as f:
        val_lines = f.readlines()

    vals = map(float, val_lines)
    return np.asarray(vals)


def draw_arrows(ax, arrows, color='k'):
    prev_arrow = arrows[0]
    for arrow in arrows[1:]:
        arrow_art = Arrow3D([prev_arrow[0], arrow[0]],
                            [prev_arrow[1], arrow[1]],
                            [prev_arrow[2], arrow[2]],
                            mutation_scale=10, lw=1.5, arrowstyle='-|>', color=color)
        ax.add_artist(arrow_art)
        prev_arrow = arrow

def plotImages(xData, yData, image, axis):
    bb = Bbox.from_bounds(xData,yData,1,1)  
    bb2 = TransformedBbox(bb,ax.transData)
    bbox_image = BboxImage(bb2,
                        norm = None,
                        origin=None,
                        clip_on=False)

    bbox_image.set_data(image)
    axis.add_artist(bbox_image)

if __name__ == '__main__':
    # svr_rbf = SVR(kernel='rbf', C=1e1, gamma=1e-1, epsilon=1e0, shrinking=True)
    svr_rbf = SVR(kernel='rbf', C=1e1, gamma=1.0)
    mds = sklearn.manifold.MDS(n_components=2)

    dataset, meta_info, all_indices = load_dataset('value_train.dat')
    coordinates = mds.fit_transform(dataset)
    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = compute_values('value_train.dat')

    all_arrows = []
    for indices in all_indices:
        arrows = []
        for x, y, z in zip(coordinates[indices, 0], coordinates[indices, 1], Z[indices]):
            arrows.append([x, y, z])
        all_arrows.append(arrows)

    # regression on [[x, y], ... ] -> [z, ...]
    train_source = [[x, y] for x, y in zip(X, Y)]

    train_target = Z

    svr_rbf.fit(train_source, train_target)

    X_test = np.linspace(np.min(X), np.max(X), num=33)
    Y_test = np.linspace(np.min(Y), np.max(Y), num=33)

    X_mesh, Y_mesh = np.meshgrid(X_test, Y_test)

    test_source = [[x, y] for x, y in zip(np.ravel(X_mesh), np.ravel(Y_mesh))]
    test_target = svr_rbf.predict(test_source)
    Z_test = test_target.reshape(X_mesh.shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    stride = (np.max(X) - np.min(X)) / 10.
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_test, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.5)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

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

    # fig = plt.figure()
    # ax = Axes3D(fig)
    ax.scatter3D(X, Y, Z, c=Z, marker='.', s=40)
    # ax.scatter3D(X, Y, Z, c=preds, marker='x')
    ax.scatter3D(centroids_x, centroids_y, centroids_z, color='black', marker='o', s=100)
    for i in range(len(centroids_x)):
        ax.text(centroids_x[i], centroids_y[i], centroids_z[i], meta_info[i], color='red', zorder=1)
        bird = plt.imread('/home/kfrankc/Desktop/bird.png')
        plotImages(centroids_x[i], centroids_y[i], bird, ax)
    # for i in range(len(all_arrows)):
    #     draw_arrows(ax, all_arrows[i])
    ax.autoscale(False)
    plt.title('MDS')
    plt.axis('tight')

    fig_polished = plt.figure()
    ax_polished = Axes3D(fig_polished)
    # ax_polished.scatter3D(X, Y, Z, c=Z, marker='.', s=40)
    surf = ax_polished.plot_surface(X_mesh, Y_mesh, Z_test, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1, antialiased=True, alpha=0.68)
    draw_arrows(ax_polished, all_arrows[0])
    # draw_arrows(ax_polished, all_arrows[1])
    # draw_arrows(ax_polished, all_arrows[2])
    # draw_arrows(ax_polished, all_arrows[3])
    # draw_arrows(ax_polished, all_arrows[4])
    # draw_arrows(ax_polished, all_arrows[5])
    # draw_arrows(ax_polished, all_arrows[6])
    # draw_arrows(ax_polished, all_arrows[1])
    fig_polished.colorbar(surf, shrink=0.5, aspect=5)
    # ax.scatter3D(X, Y, Z, c=preds, marker='x')
    plt.title('Value Landscape')
    plt.axis('tight')
    plt.show()
