import numpy as np
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from scipy.spatial import KDTree
from numpy import linalg as LA
import base64
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
import plotly.io as pio
#from pysal.lib import weights
#import pysal.lib
#from pysal.explore import esda

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
    data = anchor_points.flatten()
    # Define the spatial weights based on a contiguity rule (e.g., Queen contiguity)
    #w = weights.lat2W(anchor_points.shape[0], anchor_points.shape[1], rook=False)
    # Calculate Moran's I
    #mi = esda.moran.Moran(data, w)
    #vector[3] = mi.I

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