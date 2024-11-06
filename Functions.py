from scipy.ndimage import distance_transform_edt, generate_binary_structure, label, binary_dilation, center_of_mass
import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import skfmm
import math
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import pandas as pd
import base64
import plotly.io as pio
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy.linalg as LA
import base64
from dash.dependencies import Input, Output
import pysal.lib
from pysal.explore import esda
from pysal.lib import weights

#This file contains all of the necessary functions to run the main program
def minimum_distance_transform(image):
    # The distance_transform_edt function computes the Euclidean distance to the nearest zero (background pixel)
    # Here, we invert the image because distance_transform_edt expects the background to be zero
    dt_image = distance_transform_edt(image)
    return dt_image

def skeleton_transform(image):
    # Ensure image is in uint8 format
    image = image.astype(np.uint8) * 255
    
    # Create a skeleton container
    skel = np.zeros(image.shape, np.uint8)
    
    # Create a structuring element
    # Using MORPH_CROSS to maintain connectivity
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    
    # Repeatedly apply erosions until the image is completely eroded
    while True:
        # Erode the image
        eroded = cv.erode(image, kernel)
        
        # Open the eroded image
        temp = cv.dilate(eroded, kernel)
        
        # Subtraction to get the skeleton component of the current layer
        temp = cv.subtract(image, temp)
        
        # OR operation to add to the skeleton
        skel = cv.bitwise_or(skel, temp)
        
        # Update image to the eroded version
        image = eroded.copy()
        
        # Check if the image is completely eroded
        if cv.countNonZero(image) == 0:
            break

    return skel

def local_maxima(image):
    # Pad the image with minimum possible value on edges to handle boundary conditions
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=np.min(image) - 1)

    # Create shifted versions for all eight neighbors
    center = padded_image[1:-1, 1:-1]
    top_left = padded_image[:-2, :-2]
    top = padded_image[:-2, 1:-1]
    top_right = padded_image[:-2, 2:]
    left = padded_image[1:-1, :-2]
    right = padded_image[1:-1, 2:]
    bottom_left = padded_image[2:, :-2]
    bottom = padded_image[2:, 1:-1]
    bottom_right = padded_image[2:, 2:]

    # Compare the center pixel to all its neighbors
    local_max = ((center >= top_left) & (center >= top) & (center >= top_right) &
                 (center >= left) & (center >= right) &
                 (center >= bottom_left) & (center >= bottom) & (center >= bottom_right))

    # Create the output array of the same shape as the original image
    output = np.zeros_like(image, dtype=bool)
    output[image > 0] = local_max[image > 0]

    return output

def custom_anchors(image):
    mindist = minimum_distance_transform(image)
    locmin = local_maxima(mindist)
    anchors = np.zeros(image.shape)
    indices = np.where(image == 1)
    coordinates = list(zip(indices[0], indices[1]))
    distances = mindist[indices]

    # Sort coordinates by distances in descending order
    sorted_coords = [coord for _, coord in sorted(zip(distances, coordinates), reverse=True, key=lambda x: x[0])]

    # Initialize KD-Tree with the first point
    anchors[sorted_coords[0]] = 1
    anchor_tree = KDTree([sorted_coords[0]])

    # Iterate over sorted coordinates
    for i in range(1, len(sorted_coords)):
        point = sorted_coords[i]
        if locmin[point] == 1:  # Check if the point is a local minimum
            distance, _ = anchor_tree.query(point)
            if distance >= mindist[point]:  # Compare distance to the nearest anchor
                anchors[point] = 1
                anchor_tree = KDTree(np.vstack([anchor_tree.data, point]))  # Update KD-Tree with new anchor

    return anchors

def calculate_maximal_distances(binary_image,lp):
    # Define a connectivity that considers diagonals as well for more accurate component labeling
    connectivity = generate_binary_structure(2, 2)
    # Label connected components based on the defined connectivity
    labeled_array, num_features = label(binary_image, structure=connectivity)
    # Initialize an output array for the maximal distances
    maximal_distances = np.zeros_like(binary_image, dtype=float)
    # Iterate through each labeled component
    for i in range(1, num_features + 1):
        # Find the foreground component
        component = (labeled_array == i)
        # Find the edges of the component to calculate distance to the furthest edge
        component_edges = np.logical_xor(component, binary_dilation(component, structure=connectivity))
        
        # Calculate the distances of all foreground cells to all edge cells
        foreground_indices = np.array(np.where(component)).T
        edge_indices = np.array(np.where(component_edges)).T
        if lp == 1:
            distances = cdist(foreground_indices, edge_indices, 'cityblock')
        elif lp == 2:
            distances = cdist(foreground_indices, edge_indices, 'euclidean')
        else:
            raise ValueError('Invalid lp norm. Options are 1 or 2.')

        # Find the maximal distance to an edge for each foreground cell
        max_distances = distances.max(axis=1)
        # Assign the maximal distances back to the respective positions in the maximal_distances array
        for idx, dist in zip(foreground_indices, max_distances):
            maximal_distances[tuple(idx)] = dist

    return maximal_distances

def local_minima(image):
    local_min = np.zeros(image.shape)
    # For every non-zero pixel, check if it is a local minimum by comparing it to its 8 neighbors
    non_zero_pixels = np.array(np.where(image > 0)).T
    for pixel in non_zero_pixels:
        i, j = pixel
        # Find all neighbors within the image bounds that are non-zero
        neighbors = image[max(0, i-1):min(image.shape[0], i+2), max(0, j-1):min(image.shape[1], j+2)]
        non_zero_neighbors = neighbors[neighbors > 0]
        # Check if the pixel is a local minimum
        if len(non_zero_neighbors) != 0:
            if image[i, j] <= np.min(non_zero_neighbors):
                local_min[i, j] = 1
                image[i,j] = 1e-6
        else:
            local_min[i,j] = 1
    return local_min

def FFM_distances(grid, points):
    # Check inputs
    if not isinstance(grid, np.ndarray) or grid.dtype != np.bool_:
        raise ValueError("Grid must be a 2D binary array of type numpy.bool_.")
    if grid.ndim != 2:
        raise ValueError("Grid must be a 2D array.")

    # Label connected components of 1s in the grid
    labeled_array, num_features = label(grid)

    # Determine the object label for each point
    point_labels = {i: labeled_array[pt[0], pt[1]] for i, pt in enumerate(points)}
    if any(lab == 0 for lab in point_labels.values()):
        raise ValueError("All points must be inside an object (within 1s).")

    # Organize points by object
    objects = {}
    for point_index, obj_label in point_labels.items():
        if obj_label in objects:
            objects[obj_label].append(points[point_index])
        else:
            objects[obj_label] = [points[point_index]]

    # Initialize results list
    results = []

    # Calculate distances within the same object
    for obj_label, pts in objects.items():
        object_mask = (labeled_array == obj_label)
        
        # Calculate distance map for one point in the component
        src = pts[0]
        distance = np.full(grid.shape, np.inf)
        distance[src[0], src[1]] = 0  # Seed point
        phi = np.ma.MaskedArray(distance, mask=~object_mask)
        distance_map = skfmm.distance(phi)  # Apply FMM from the source point

        # Record distances between all pairs of points within the same object
        for src in pts:
            for tgt in pts:
                if src != tgt:  # Skip self-distance
                    dist = distance_map[tgt[0], tgt[1]]
                    if not np.isinf(dist) and dist != 0:
                        results.append((src, tgt, dist, obj_label))

    # Set distances to inf for points in different objects
    for i, pt1 in enumerate(points):
        for j, pt2 in enumerate(points):
            if i != j and point_labels[i] != point_labels[j]:
                results.append((pt1, pt2, np.inf, point_labels[i]))
                results.append((pt2, pt1, np.inf, point_labels[j]))

    # Remove all inf distances
    results = [item for item in results if not np.isinf(item[2])]

    return results

def create_mst_and_calculate_percentile(results, percentile=80):
    # Organize distances by object label
    objects = {}
    for src, tgt, dist, obj_label in results:
        if obj_label in objects:
            objects[obj_label].append((src, tgt, dist))
        else:
            objects[obj_label] = [(src, tgt, dist)]

    percentiles = []

    # Process each object's distances
    for obj_label, distances in objects.items():
        # Extract unique points in the object
        points = list(set([src for src, _, _ in distances] + [tgt for _, tgt, _ in distances]))
        num_points = len(points)
        point_indices = {point: i for i, point in enumerate(points)}

        # Create distance matrix for the current object
        dist_matrix = np.full((num_points, num_points), np.inf)
        for src, tgt, dist in distances:
            i = point_indices[src]
            j = point_indices[tgt]
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

        # Compute MST
        mst = minimum_spanning_tree(dist_matrix).toarray()

        # Collect traversal distances from the MST
        traversal_distances = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    traversal_distances.append(mst[i, j] if mst[i, j] > 0 else mst[j, i])

        if traversal_distances:
            # Calculate the desired percentile
            percentile_value = np.percentile(traversal_distances, percentile)
            percentiles.append(percentile_value)

    # Return the average percentile value across all objects
    if percentiles:
        average_percentile = np.mean(percentiles)
    else:
        average_percentile = None

    return percentiles

def oscar_mike_vec(binary_image, anchor_type = 'separate'):
    """
    This function calculates the Oscar metric for a binary image.
    Input:
    binary_image: a 2D numpy array of shape (n, m) containing the binary image
    lp: the lp norm to use for the distance calculation
    anchor_type: the type of anchor to use. Can be 'separate' or 'mindist'
    """
    # First we do the minimal distance transform of the image
    anchor_points = np.zeros_like(binary_image)
    if anchor_type == 'separate':
        #blablabla
        anchor_points = custom_anchors(binary_image)
    elif anchor_type == 'mindist':
        dt_image_min = minimum_distance_transform(binary_image)
        locmax = local_maxima(dt_image_min)
        dt_image_max,_,_ = calculate_maximal_distances(locmax)
        anchor_points = local_minima(dt_image_max)
    else:
        raise ValueError('Invalid anchor_type. Options are "separate" or "mindist"')
    
    vector = np.zeros(3)
    #vector[0] = np.sum(binary_image)
    white_pixel_ratio = np.sum(binary_image) / (binary_image.shape[0] * binary_image.shape[1])
    intensity = np.sum(anchor_points)/white_pixel_ratio
    vector[0] = intensity
    # Define a connectivity that considers diagonals as well for more accurate component labeling
    connectivity = generate_binary_structure(2, 2)
    # Label connected components based on the defined connectivity
    labeled_array, num_features = label(binary_image, structure=connectivity)
    comp_anchors = []
    anchors = [(np.where(anchor_points)[0][i],np.where(anchor_points)[1][i]) for i in range(len(np.where(anchor_points)[0]))]
    for j in range(1,num_features+1):
        component = [(np.where(labeled_array==j)[0][i],np.where(labeled_array==j)[1][i]) for i in range(len(np.where(labeled_array==j)[0]))]
        # Find the number of anchor points in the component
        set1 = set(anchors)
        set2 = set(component)
        common = list(set1.intersection(set2))
        comp_anchors.append(common)
    comp_anchor_size = [len(comp_anchors[i]) for i in range(len(comp_anchors))]
    comp_anchor_variance = np.var(comp_anchor_size)
    vector[1] = comp_anchor_variance
    dists = calculate_distance_among_objects_sped_up(binary_image.astype(bool),anchors)
    finite_distances = [value for value in dists.values() if math.isfinite(value)]
    average_distance = sum(finite_distances) / len(finite_distances)
    vector[2] = average_distance / (binary_image.shape[0] * binary_image.shape[1])
    return vector
    
def oscar_ragnar_vec(binary_image):
    vector = np.zeros(3)
    intensity = np.sum(binary_image) / (binary_image.shape[0] * binary_image.shape[1])
    vector[0] = intensity
    connectivity = generate_binary_structure(2, 2)
    # Label connected components based on the defined connectivity
    labeled_array, num_features = label(binary_image, structure=connectivity)
    vector[1] = num_features 
    anchor_points = custom_anchors(binary_image)
    vector[2] = np.sum(anchor_points)
    return vector

def count_normalized_connection_lengths(binary_image):
    n_rows, n_cols = binary_image.shape
    
    # Function to count contiguous 1s in a row or column
    def count_contiguous_ones(arr, total_length):
        lengths = []
        count = 0
        for val in arr:
            if val == 1:
                count += 1
            else:
                if count > 0:
                    lengths.append(count / total_length)  # Normalize by total length
                    count = 0
        if count > 0:
            lengths.append(count / total_length)  # Normalize by total length
        return lengths
    
    # Scan horizontally
    horizontal_lengths = []
    for row in binary_image:
        horizontal_lengths.extend(count_contiguous_ones(row, n_cols))
    
    # Scan vertically
    vertical_lengths = []
    for col in binary_image.T:
        vertical_lengths.extend(count_contiguous_ones(col, n_rows))
    
    # Combine horizontal and vertical lengths
    all_lengths = horizontal_lengths + vertical_lengths
    
    # Calculate average normalized length
    if all_lengths:
        average_normalized_length = sum(all_lengths) / len(all_lengths)
    else:
        average_normalized_length = 0.0
    
    return average_normalized_length

def oscar_oscar_vec(binary_image):
    vector = np.zeros(3)
    intensity = np.sum(binary_image)/(binary_image.shape[0]*binary_image.shape[1])
    vector[0] = intensity
    # Define a connectivity that considers diagonals as well for more accurate component labeling
    connectivity = generate_binary_structure(2, 2)
    # Label connected components based on the defined connectivity
    labeled_array, num_features = label(binary_image, structure=connectivity)
    comp_anchors = []
    anchor_points = custom_anchors(binary_image)
    anchors = [(np.where(anchor_points)[0][i],np.where(anchor_points)[1][i]) for i in range(len(np.where(anchor_points)[0]))]
    for j in range(1,num_features+1):
        component = [(np.where(labeled_array==j)[0][i],np.where(labeled_array==j)[1][i]) for i in range(len(np.where(labeled_array==j)[0]))]
        # Find the number of anchor points in the component
        set1 = set(anchors)
        set2 = set(component)
        common = list(set1.intersection(set2))
        comp_anchors.append(common)
    comp_anchor_size = [len(comp_anchors[i]) for i in range(len(comp_anchors))]
    comp_anchor_variance = np.var(comp_anchor_size)
    vector[1] = comp_anchor_variance
    vector[2] = count_normalized_connection_lengths(binary_image)
        
    return vector

def oscar_micky_vec(binary_image,alternative = 0):
    vector = np.zeros(3)
    anchor_points = custom_anchors(binary_image)
    connectivity = generate_binary_structure(2, 2)
    labeled_array, num_features = label(binary_image, structure=connectivity)
    num_ancs = np.zeros(num_features)
    rel_areas = np.zeros(num_features)
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        nr_anchors = np.sum(anchor_points[component])
        rel_area = np.sum(component) / (binary_image.shape[0] * binary_image.shape[1])
        rel_areas[i-1] = rel_area
        num_ancs[i-1] = nr_anchors
    vector[0] = np.mean(num_ancs/rel_areas)
    vector[1] = np.var(num_ancs/rel_areas)
    if alternative == 0:
        anchors = [(np.where(anchor_points)[0][i],np.where(anchor_points)[1][i]) for i in range(len(np.where(anchor_points)[0]))]
        FFM_dist = FFM_distances(binary_image.astype(bool), anchors)
        FFM_scaled = []
        for src, tgt, dist, obj_label in FFM_dist:
            FFM_scaled.append((src,tgt,dist/(binary_image.shape[0] * binary_image.shape[1]),obj_label))
        vector[2] = np.mean(create_mst_and_calculate_percentile(FFM_scaled))
        print('The correlation feature is the average percentile of the MST traversal distances')
    elif alternative == 1:
        vector[2] = np.sum(anchor_points)
        print('The correlation feature is the nr of anchor points')
    elif alternative == 2:
        vector[2] = np.sum(binary_image)/(binary_image.shape[0]*binary_image.shape[1])
        print('The correlation feature is the intensity')
    elif alternative == 3:
        vector[2] = num_features
        print('The correlation feature is the number of components')
    elif alternative == 4:
        # Flatten the grid to create a 1D array
        data = anchor_points.flatten()
        # Define the spatial weights based on a contiguity rule (e.g., Queen contiguity)
        w = weights.lat2W(anchor_points.shape[0], anchor_points.shape[1], rook=False)
        # Calculate Moran's I
        mi = esda.moran.Moran(data, w)
        vector[2] = mi.I
        print('The correlation feature is the Moran\'s I')
    else:
        raise ValueError('Invalid alternative. Options are 0, 1, 2, 3, 4')
    return vector
    
def skeleton_anchors(image):
    skel = skeleton_transform(image)
    # Assuming 'skel' is your skeletonized binary image array
    labeled_array, num_features = label(skel)

    # Calculate centroids for each component
    centroids = center_of_mass(skel, labeled_array, range(1, num_features + 1))

    return centroids

def plot_weight_distribution(matrices, title):
    plt.figure(figsize=(10, 6))
    for idx, matrix in enumerate(matrices):
        weights = matrix[matrix > 0]  # Only consider non-zero weights
        plt.hist(weights, bins=50, alpha=0.5, label=f'Matrix {idx+1}')
    plt.title(f'Weight Distribution in {title}')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_degree_distribution(matrices, title):
    plt.figure(figsize=(10, 6))
    for idx, matrix in enumerate(matrices):
        degrees = np.sum(matrix, axis=0)  # Sum of weights per node (degree)
        plt.hist(degrees, bins=50, alpha=0.5, label=f'Matrix {idx+1}')
    plt.title(f'Degree Distribution in {title}')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_eigenvalue_distribution(matrices, title):
    plt.figure(figsize=(10, 6))
    for idx, matrix in enumerate(matrices):
        # Compute the graph Laplacian
        D = np.diag(np.sum(matrix, axis=1))
        L = D - matrix
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)
        plt.hist(eigenvalues, bins=50, alpha=0.5, label=f'Matrix {idx+1}')
    plt.title(f'Eigenvalue Distribution of Laplacians in {title}')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def build_adjacency_matrix(points, distances):
    # Initialize the adjacency matrix
    n = len(points)
    adjacency_matrix = np.zeros((n, n))
    
    # Fill in the adjacency matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                adjacency_matrix[i, j] = 0  # No self-loops, assuming 0 distance to self is not useful
            else:
                distance_key = f"Distance from {points[i]} to {points[j]}"
                distance = distances.get(distance_key, np.inf)  # Default to inf if not found
                if distance != np.inf:
                    adjacency_matrix[i, j] = 1 / distance  # Inverse of the distance
                else:
                    adjacency_matrix[i, j] = 0  # Zero for inf distances
    
    return adjacency_matrix

def scatter_plot_with_images(osc_vecs, image_paths, datanames, filename='interactive_plot.html'):
    def encode_image(image_path):
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    df_full = []
    some_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(len(osc_vecs)):
        x_values = np.array(osc_vecs[i])[:, 0]
        y_values = np.array(osc_vecs[i])[:, 1]
        z_values = np.array(osc_vecs[i])[:, 2]
        encoded_images = [encode_image(image_path) for image_path in image_paths[i]]
        df = pd.DataFrame({'x': x_values, 'y': y_values, 'z': z_values, 'image': [f"data:image/png;base64,{img}" for img in encoded_images], 'color': some_colors[i]})
        df['id'] = df.index + len(df_full[i - 1]['id']) if i > 0 else df.index
        df['dataname'] = datanames[i]
        df_full.append(df)

    df_all = pd.concat(df_full, ignore_index=True)

    fig = go.Figure()

    for color in df_all['color'].unique():
        df_subset = df_all[df_all['color'] == color]
        trace_name = df_subset['dataname'].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=df_subset['x'],
            y=df_subset['y'],
            z=df_subset['z'],
            mode='markers',
            marker=dict(size=5, color=color),
            customdata=df_subset['id'],  # Use customdata to store the id
            name=trace_name,
        ))

    # Turn off native plotly.js hover effects
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='intensity', showgrid=True, zeroline=True),
            yaxis=dict(title='variance', showgrid=True, zeroline=True),
            zaxis=dict(title='correlation', showgrid=True, zeroline=True),
        ),
        title='3D feature vectors with images',
    )

    app = JupyterDash(__name__)
    server = app.server

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        [Output("graph-tooltip", "show"),
         Output("graph-tooltip", "bbox"),
         Output("graph-tooltip", "children")],
        [Input("graph-basic-2", "hoverData")],
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, {}, ""

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        idx = pt["customdata"]  # Get the index from customdata

        df_row = df_all.iloc[idx]
        img_data = df_row['image']

        children = [
            html.Div([
                html.Img(src=img_data, style={"width": "100%"}),
            ], style={'width': '100px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    pio.write_html(fig, file=filename)
    # Run the Dash app locally
    portnr = np.random.randint(8000, 9000)
    app.run_server(port=portnr)
    return portnr
    
def coordinates_for_bounding_box(data):
    means = np.mean(data, axis=1)
    cov = np.cov(data)
    eval, evec = LA.eig(cov)
    centered_data = data - means[:, np.newaxis]
    aligned_coords = np.matmul(evec.T, centered_data)
    xmin, xmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :])
    ymin, ymax = np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :])
    zmin, zmax = np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])
    
    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([
        [x1, x1, x2, x2, x1, x1, x2, x2],
        [y1, y2, y2, y1, y1, y2, y2, y1],
        [z1, z1, z1, z1, z2, z2, z2, z2]
    ])
    
    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis]
    
    volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    
    return rrc, volume

def scatter_plot_with_images_update(osc_vecs, image_paths, datanames, filename='interactive_plot.html'):
    def encode_image(image_path):
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    df_full = []
    some_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i in range(len(osc_vecs)):
        x_values = np.array(osc_vecs[i])[:, 0]
        y_values = np.array(osc_vecs[i])[:, 1]
        z_values = np.array(osc_vecs[i])[:, 2]
        encoded_images = [encode_image(image_path) for image_path in image_paths[i]]
        df = pd.DataFrame({
            'x': x_values, 
            'y': y_values, 
            'z': z_values, 
            'image': [f"data:image/png;base64,{img}" for img in encoded_images], 
            'color': some_colors[i]
        })
        df['id'] = df.index + (len(df_full[i - 1]['id']) if i > 0 else 0)
        df['dataname'] = datanames[i]
        df_full.append(df)

    df_all = pd.concat(df_full, ignore_index=True)

    # Normalize the coordinates
    min_vals = df_all[['x', 'y', 'z']].min()
    max_vals = df_all[['x', 'y', 'z']].max()
    df_all[['x', 'y', 'z']] = (df_all[['x', 'y', 'z']] - min_vals) / (max_vals - min_vals)

    # Compute the minimal bounding box
    points = df_all[['x', 'y', 'z']].values.T
    bounding_box, volume = coordinates_for_bounding_box(points)

    fig = go.Figure()

    for color in df_all['color'].unique():
        df_subset = df_all[df_all['color'] == color]
        trace_name = df_subset['dataname'].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=df_subset['x'],
            y=df_subset['y'],
            z=df_subset['z'],
            mode='markers',
            marker=dict(size=5, color=color),
            customdata=df_subset['id'],  # Use customdata to store the id
            name=trace_name,
        ))

    # Add the bounding box to the plot
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
    ]
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[bounding_box[0, edge[0]], bounding_box[0, edge[1]]],
            y=[bounding_box[1, edge[0]], bounding_box[1, edge[1]]],
            z=[bounding_box[2, edge[0]], bounding_box[2, edge[1]]],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False
        ))

    # Turn off native plotly.js hover effects
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='intensity', showgrid=True, zeroline=True),
            yaxis=dict(title='variance', showgrid=True, zeroline=True),
            zaxis=dict(title='correlation', showgrid=True, zeroline=True),
        ),
        title=f'3D feature vectors with images - Bounding Box Volume: {volume:.2f}',
    )

    app = Dash(__name__)
    server = app.server

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        [Output("graph-tooltip", "show"),
         Output("graph-tooltip", "bbox"),
         Output("graph-tooltip", "children")],
        [Input("graph-basic-2", "hoverData")],
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, {}, ""

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        idx = pt["customdata"]  # Get the index from customdata

        df_row = df_all.iloc[idx]
        img_data = df_row['image']

        children = [
            html.Div([
                html.Img(src=img_data, style={"width": "100%"}),
            ], style={'width': '100px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    pio.write_html(fig, file=filename)
    # Run the Dash app locally
    portnr = np.random.randint(8000, 9000)
    app.run_server(port=portnr)
    return portnr, volume