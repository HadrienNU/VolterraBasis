import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay

# import meshio # Library for mesh IO
# TODO Implement la même function mais en sparse puisque une grande partie des valeurs vont être zéros


def uniform_line(x_start, x_end, num_elements):
    """
    Creates a 1D uniform grid.

    Args:
        x_start: Leftmost x-coordinate of the domain.
        x_end: Rightmost x-coordinate of the domain.
    """

    return np.linspace(x_start, x_end, num_elements + 1)


def non_uniform_line(x_start, x_end, num_elements, ratio):
    """
    Creates a 1D non-uniform grid by placing num_elements nodes in
    geometric progression between x_start and x_end with ratio ratio.

    Args:
        x_start: This is the first param.
        x_end: This is a second param.
    """

    # Create grid points between 0 and 1
    h = (ratio - 1) / (ratio ** num_elements - 1)
    x = np.append([0], h * np.cumsum(ratio ** np.arange(num_elements)))

    return x_start + x * (x_end - x_start)


def data_driven_line(data, bins=10, x_start=None, x_end=None):
    """
    Creates a 1D grid using histogram estimation.

    Args:
        data: Data points
        x_start: Leftmost x-coordinate of the domain.
        x_end: Rightmost x-coordinate of the domain.

    Returns:
        A fully initialized instance of Mesh.
    """
    if x_start is None:
        x_start = data.min()
    if x_end is None:
        x_end = data.max()
    return np.histogram_bin_edges(data, bins=bins, range=(x_start, x_end))


def centroid_driven_line(data, bins=100):
    """
    Creates a mesh line based on centroids of the data, to get more cell around point with more datas

    Args:
        data: Data points, we already assume that we only have reactive trajectory
        bins: wanted number of element
    """
    kmeans = KMeans(n_clusters=bins, random_state=0).fit(data)
    # For 1D, we ravel and sort the cluster center
    # We have to add the boundary
    mesh = np.concatenate((np.array([data.min()]), np.sort(kmeans.cluster_centers_.ravel()), np.array([data.max()])))
    return mesh


def centroid_driven_mesh(data, bins=100, boundary_vertices=None):
    """
    Creates a mesh line based on centroids of the data, to get more cell around point with more datas

    Args:
        data: Data points, we already assume that we only have reactive trajectory
        bins: wanted number of element
    """
    # Find clusters
    kmeans = KMeans(n_clusters=bins, random_state=0).fit(data)
    # We have to add the boundary, let's take the convex hull of the data if not defined
    if boundary_vertices is None:
        hull = ConvexHull(data)
        boundary_vertices = data[hull.vertices]
    vertices = np.concatenate((boundary_vertices, kmeans.cluster_centers_))
    # For ND, do Delaunay triangulation of the space
    tri = Delaunay(vertices)
    return vertices, tri.simplices


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    points = np.random.rand(5000, 2)
    vertices, tri = centroid_driven_mesh(points, 10, boundary_vertices=[[0, 0], [0, 1], [1, 0], [1, 1]])
    plt.plot(points[:, 0], points[:, 1], "x")
    plt.triplot(vertices[:, 0], vertices[:, 1], tri)
    plt.plot(vertices[:, 0], vertices[:, 1], "o")
    plt.show()
