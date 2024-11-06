import numpy as np
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure, find_objects
from scipy.spatial import KDTree
#from pysal.lib import weights
#import pysal.lib
#from pysal.explore import esda
import base64
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.io as pio
from numpy import linalg as LA
#import pointpats as pp
from scipy.ndimage import maximum_filter, generate_binary_structure
from sklearn.neighbors import BallTree
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from re import sub
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from scipy.spatial import KDTree

def expected_nearest_neighbor_distance(n_points, area):
    # Expected nearest neighbor distance in a Poisson random distribution
    return 1 / (2 * np.sqrt(n_points / area))

def compute_metric(coordinates, k, area_size=(1, 1)):
    # Number of points
    n_points = len(coordinates)

    # Calculate total area of the grid
    total_area = area_size[0] * area_size[1]

    # Calculate the expected nearest neighbor distance
    r_expected = expected_nearest_neighbor_distance(n_points, total_area)

    # Compute the search radius as a multiple of the expected nearest neighbor distance
    search_radius = k * r_expected

    # Calculate point density
    point_density = n_points / total_area

    # Expected number of points in the search radius area
    search_area = np.pi * search_radius**2
    expected_count = point_density * search_area

    # Build KDTree for fast radius-based searches
    tree = KDTree(coordinates)

    # Get the number of points within the search radius for each point
    counts = []
    for point in coordinates:
        indices = tree.query_ball_point(point, search_radius)
        count = len(indices) - 1  # Exclude the point itself
        counts.append(count)

    # Compute the empirical mean and variance of the counts
    counts = np.array(counts)
    empirical_mean = np.mean(counts)
    empirical_variance = np.var(counts)

    # Theoretical variance for a Poisson distribution (equals the mean)
    theoretical_variance = expected_count

    # Compute the metric (empirical variance - theoretical variance) / (empirical variance + theoretical variance)
    if empirical_variance + theoretical_variance > 0:
        metric = (empirical_variance - theoretical_variance) / (empirical_variance + theoretical_variance)
    else:
        metric = 0

    return metric

def minimum_distance_transform(image):
    # The distance_transform_edt function computes the Euclidean distance to the nearest zero (background pixel)
    # Here, we invert the image because distance_transform_edt expects the background to be zero
    dt_image = distance_transform_edt(image)
    return dt_image

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
    anchor_radii = np.zeros(image.shape)
    anchor_label = np.zeros(image.shape)
    indices = np.where(image == 1)
    coordinates = list(zip(indices[0], indices[1]))
    distances = mindist[indices]
    connectivity = generate_binary_structure(2, 2)
    labeled_array, num_features = label(image, structure=connectivity)

    # Sort coordinates by distances in descending order
    sorted_coords = [coord for _, coord in sorted(zip(distances, coordinates), reverse=True, key=lambda x: x[0])]
    # Initialize KD-Tree with the first point
    anchors[sorted_coords[0]] = 1
    anchor_radii[sorted_coords[0]] = mindist[sorted_coords[0]]
    anchor_label[sorted_coords[0]] = labeled_array[sorted_coords[0]]
    anchor_tree = KDTree([sorted_coords[0]])

    # Iterate over sorted coordinates
    for i in range(1, len(sorted_coords)):
        point = sorted_coords[i]
        if locmin[point] == 1:  # Check if the point is a local minimum
            distance, _ = anchor_tree.query(point)
            if distance >= mindist[point]:  # Compare distance to the nearest anchor
                anchors[point] = 1
                anchor_radii[point] = mindist[point]
                anchor_label[point] = labeled_array[point]
                anchor_tree = KDTree(np.vstack([anchor_tree.data, point]))  # Update KD-Tree with new anchor
    return anchors, anchor_radii, anchor_label

def optimized_morans_I(binary_image):
    # Convert the binary image to a numpy array if it's not already
    if not isinstance(binary_image, np.ndarray):
        binary_image = np.array(binary_image)
    
    # Get the dimensions of the image
    rows, cols = binary_image.shape
    
    # Compute the mean of the binary image
    mean_value = np.mean(binary_image)
    
    # Compute the numerator and denominator for Moran's I
    numerator = 0
    denominator = 0
    W = 0  # Total number of valid neighbor pairs
    
    for i in range(rows):
        for j in range(cols):
            # Current cell value
            val = binary_image[i, j]
            
            # Compute the deviation from the mean
            deviation = val - mean_value
            
            # Add to denominator
            denominator += deviation ** 2
            
            # Check 4-nearest neighbors and update numerator and weight sum
            if i > 0:  # above
                W += 1
                numerator += deviation * (binary_image[i-1, j] - mean_value)
            if i < rows - 1:  # below
                W += 1
                numerator += deviation * (binary_image[i+1, j] - mean_value)
            if j > 0:  # left
                W += 1
                numerator += deviation * (binary_image[i, j-1] - mean_value)
            if j < cols - 1:  # right
                W += 1
                numerator += deviation * (binary_image[i, j+1] - mean_value)
    
    # Calculate Moran's I
    morans_I = (rows * cols / W) * (numerator / denominator)
    
    return morans_I

def oscar_micky_vec(binary_image):
    vector = np.zeros(4)
    anchor_points = custom_anchors(binary_image)
    vector[0] = np.sum(anchor_points) / (np.sum(binary_image) / (binary_image.shape[0] * binary_image.shape[1]))
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
    vector[1] = np.mean(num_ancs/rel_areas)
    vector[2] = np.var(num_ancs/rel_areas)
    # Flatten the grid to create a 1D array
    vector[3] = optimized_morans_I(anchor_points)
    return vector

def minimum_distance_transform_optim(image):
    dt_image = distance_transform_edt(image)
    return dt_image

def local_maxima_optim(image):
    footprint = generate_binary_structure(2, 2)
    max_filter_image = maximum_filter(image, footprint=footprint)
    local_max = (image == max_filter_image)
    return local_max

def custom_anchors_optim(image):
    mindist = minimum_distance_transform_optim(image)
    locmin = local_maxima_optim(mindist)
    anchors = np.zeros(image.shape)
    indices = np.where(image == 1)
    coordinates = np.array(list(zip(indices[0], indices[1])))
    distances = mindist[indices]

    sorted_indices = np.argsort(distances)[::-1]
    sorted_coords = coordinates[sorted_indices]

    anchors[tuple(sorted_coords[0])] = 1
    anchor_tree = BallTree(sorted_coords[:1])

    for i in range(1, len(sorted_coords)):
        point = sorted_coords[i]
        if locmin[tuple(point)] == 1:
            distance, _ = anchor_tree.query(point.reshape(1, -1))
            if distance >= mindist[tuple(point)]:
                anchors[tuple(point)] = 1
                anchor_tree = BallTree(sorted_coords[:i+1])

    return anchors

def optimized_morans_I_optim(binary_image):
    mean_value = np.mean(binary_image)
    deviation = binary_image - mean_value
    denominator = np.sum(deviation ** 2)

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbor_sum = convolve2d(binary_image, kernel, mode='same', boundary='fill', fillvalue=0)
    numerator = np.sum(deviation * (neighbor_sum - mean_value))

    W = np.sum(kernel)
    rows, cols = binary_image.shape
    morans_I = (rows * cols / W) * (numerator / denominator)

    return morans_I

def oscar_micky_vec_optim(binary_image):
    vector = np.zeros(4)
    anchor_points = custom_anchors_optim(binary_image)
    vector[0] = np.sum(anchor_points) / (np.sum(binary_image) / (binary_image.shape[0] * binary_image.shape[1]))
    connectivity = generate_binary_structure(2, 2)
    labeled_array, num_features = label(binary_image, structure=connectivity)
    num_ancs = np.bincount(labeled_array.ravel())[1:]
    rel_areas = np.bincount(labeled_array.ravel(), weights=binary_image.ravel())[1:] / (binary_image.shape[0] * binary_image.shape[1])
    vector[1] = np.mean(num_ancs/rel_areas)
    vector[2] = np.var(num_ancs/rel_areas)
    vector[3] = optimized_morans_I_optim(anchor_points)
    return vector

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

def find_local_maxima(array, allow_ties=True):
    # Step 1: Calculate the distance transform
    distance = distance_transform_edt(array)
    # Step 2: Find local maximas using an 8-neighborhood structure
    neighborhood = np.ones((3, 3), dtype=bool)
    local_max = (maximum_filter(distance, footprint=neighborhood) == distance)
    # Handle ties
    if not allow_ties:
        local_max &= (maximum_filter(distance, footprint=neighborhood, mode='constant', cval=-np.inf) < distance)

    # Step 3: Calculate the distance to the nearest False cell (radius) for each local maxima
    labels, num_features = label(array)
    objects = find_objects(labels)
    
    radii = np.zeros_like(distance)
    for i, slice_ in enumerate(objects, start=1):
        mask = (labels[slice_] == i)
        local_max_in_object = local_max[slice_] & mask
        distances = distance[slice_][local_max_in_object]
        radii[slice_][local_max_in_object] = distances

    # Step 4: Filter out overlapping maximas within the same connected object
    unique_maximas = []
    for i, slice_ in enumerate(objects, start=1):
        mask = (labels[slice_] == i)
        local_max_in_object = np.argwhere(local_max[slice_] & mask)
        
        if local_max_in_object.size == 0:
            continue
        
        maximas = []
        radii_list = []
        for loc in local_max_in_object:
            loc = tuple(loc)
            radius = radii[slice_][loc]
            overlap = any(np.linalg.norm(np.array(loc) - np.array(m)) <= r + radius for m, r in zip(maximas, radii_list))
            if not overlap:
                maximas.append(loc)
                radii_list.append(radius)
        
        for loc, radius in zip(maximas, radii_list):
            if radius > 0:
                unique_maximas.append((slice_[0].start + loc[0], slice_[1].start + loc[1], radius))
    
    return unique_maximas

def calculate_statistics(counts):
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    return mean_count, var_count

def count_ones_in_subgrids(binary_array, subgrid_size=10):
    rows, cols = binary_array.shape
    counts = []
    for i in range(0, rows, subgrid_size):
        for j in range(0, cols, subgrid_size):
            subgrid = binary_array[i:i+subgrid_size, j:j+subgrid_size]
            counts.append(np.sum(subgrid))
    return np.array(counts)

def construct_metric(binary_array, subgrid_size=10):
    counts = count_ones_in_subgrids(binary_array, subgrid_size)
    mean_count, var_count = calculate_statistics(counts)
    
    # Expected mean and variance for a random distribution
    total_cells = subgrid_size * subgrid_size
    p = np.mean(binary_array)  # Proportion of 1s in the entire array
    expected_variance = total_cells * p 
    
    # Normalize the variance to be between -1 and 1
    metric = (var_count - expected_variance) / (var_count + expected_variance)
    # normalize the variance again to be between 0 and 1
    metric = (metric + 1) / 2
    
    return metric

def oscar_ragnar_vec(image):
    local_max_with_radii = find_local_maxima(image, allow_ties=True)
    vec = np.zeros(4)
    grid_size = image.shape[0]*image.shape[1]
    vec[0] = len(local_max_with_radii)
    vec[1] = np.mean([x[2]/grid_size for x in local_max_with_radii])
    vec[2] = np.std([x[2]/grid_size for x in local_max_with_radii])
    anchor_pic = np.zeros_like(image)
    for x, y, r in local_max_with_radii:
        anchor_pic[x, y] = 1
    vec[3] = construct_metric(anchor_pic)
    return vec

def oscar_vector(image):
    local_max_with_radii = find_local_maxima(image, allow_ties=True)
    vec = np.zeros(4)
    grid_size = image.shape[0]*image.shape[1]
    # First feature is the white pixel ratio
    vec[0] = np.sum(image)/grid_size
    # Second feature is the average fill rate of radius
    connectivity = generate_binary_structure(2, 2)
    labeled_array, num_features = label(image, structure=connectivity)
    fill_rates = np.zeros(len(local_max_with_radii))
    anchor_pic = np.zeros_like(image)
    for i in range(len(local_max_with_radii)):
        x, y, r = local_max_with_radii[i]
        component = (labeled_array == labeled_array[x, y])
        fill_rates[i] = r/np.sum(component)
        anchor_pic[x, y] = 1
    vec[1] = np.mean(fill_rates)
    vec[2] = np.var(fill_rates) * 4
    # Fourth feature is the custom metric
    vec[3] = construct_metric(anchor_pic)
    return vec

def supervector(image):
    supervector = np.zeros(7)
    # First feature is the white pixel ratio
    supervector[0] = np.sum(image)/(image.shape[0]*image.shape[1])
    # Second feature is the nr of objects / maximum nr of objects
    connectivity = generate_binary_structure(2, 2)
    labeled_array, num_features = label(image, structure=connectivity)
    supervector[1] = 9*num_features / (image.shape[0]*image.shape[1])
    # Third feature is the average number of anchor points per object
    local_max_with_radii = find_local_maxima(image, allow_ties=True)
    anchor_image = np.zeros_like(image)
    fill_rates = np.zeros(len(local_max_with_radii))
    for j in range(len(local_max_with_radii)):
        x, y, r = local_max_with_radii[j]
        anchor_image[x, y] = 1
        component = (labeled_array == labeled_array[x, y])
        fill_rates[j] = r/np.sum(component)
    rel_nr_anchors = np.zeros(num_features)
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        rel_nr_anchors[i-1] = np.sum(anchor_image[component]) / np.sum(component)
    supervector[2] = np.mean(rel_nr_anchors)
    # Fourth feature is the variance of the number of anchor points per object
    supervector[3] = np.var(rel_nr_anchors)*4
    # Fifth feature is the average fill rate of radius
    supervector[4] = np.mean(fill_rates)
    # Sixth feature is the variance of the fill rate of radius
    supervector[5] = np.var(fill_rates)*4
    # Seventh feature is the custom metric
    supervector[6] = construct_metric(anchor_image)
    return supervector

def plotting_vecs(all_vecs,labels,datanames,some_colors,subplot_dim):
    fig, axes = plt.subplots(subplot_dim, subplot_dim, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    
    # Plot the data in matrix form
    for i in range(subplot_dim):
        for j in range(subplot_dim):
            if i == j:
                for k in range(len(all_vecs)):
                    axes[i, j].hist(np.array(all_vecs[k])[:,i],label=datanames[k],color=some_colors[k],alpha=0.5)
                    #axes[i, j].legend()
            else:
                for k in range(len(all_vecs)):
                    axes[i, j].scatter(np.array(all_vecs[k])[:,j], np.array(all_vecs[k])[:,i],label=datanames[k],color=some_colors[k],alpha=0.5)
                    #axes[i, j].legend()
            if i == 3:
                axes[i, j].set_xlabel(labels[j])
            if j == 0:
                axes[i, j].set_ylabel(labels[i])

    # Create a new figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(10, 2))  # Adjust the figure size to accommodate the horizontal legend

    # Create legend handles manually
    handles = [mpatches.Patch(color=some_colors[i], label=datanames[i]) for i in range(len(datanames))]

    # Add the legend to the subplot
    ax_legend.legend(handles=handles, loc='center', ncol=len(handles))

    # Hide the axes
    ax_legend.axis('off')

    # Show the plots
    plt.show()

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
    df_all['id'] = df_all.index
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

def comparison(vec1,vec2):
    comp = []
    for i in range(len(vec1)):
        calc = (vec1[i]-vec2[i])**2/np.max([vec1[i],vec2[i],vec1[i]-vec2[i]])**2
        comp.append(calc)
    comp = np.array(comp)
    return np.mean(comp)