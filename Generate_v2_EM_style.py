# This is version 2
# Instead of ISM + SH_Directivity + Shoebox, we use Ray Tracing + Spherical Array + Complex Geometries here
# Has complex geometries
import numpy as np
import pandas as pd
from scipy.io import wavfile
import pyroomacoustics as pra
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon, Point
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D
import os
import multiprocessing

# Global Constants
#TODO: frequency band error, remove sources after done
LOOKUP_TABLE = [
    'hard_surface', 'brickwork', 'rough_concrete', 'unpainted_concrete', 
    'rough_lime_wash', 'smooth_brickwork_flush_pointing', 'smooth_brickwork_10mm_pointing', 
    'brick_wall_rough', 'ceramic_tiles', 'limestone_wall', 'reverb_chamber', 
    'plasterboard', 'wooden_lining', 'glass_3mm', 'glass_window',
    'double_glazing_30mm', 'double_glazing_10mm', 'wood_1.6cm', 'curtains_cotton_0.5',
    'curtains_0.2', 'curtains_velvet', 'curtains_glass_mat', 'carpet_cotton', 
    'carpet_6mm_closed_cell_foam', 'carpet_6mm_open_cell_foam', 'carpet_tufted_9m',
    'felt_5mm', 'carpet_hairy', 'concrete_floor', 'marble_floor', 'orchestra_1.5_m2',
    'panel_fabric_covered_6pcf', 'panel_fabric_covered_8pcf', 'ceiling_fibre_absorber'
]
WALLS = LOOKUP_TABLE[:22]
FLOOR = np.concatenate((LOOKUP_TABLE[4:8], LOOKUP_TABLE[22:31]))
CEIL = np.concatenate((LOOKUP_TABLE[:5], LOOKUP_TABLE[31:]))

# Utility Functions
def get_random_dimensions():
    """
    Generate random dimensions for the room.
    Returns:
        tuple: Random dimensions (Lx, Ly, Lz).
    """
    dimensions_range = np.arange(2, 8, 1)
    return tuple(np.random.choice(dimensions_range, size=1)[0] for _ in range(3))


def get_random_positions(Lx, Ly, Lz):
    """
    Generate random positions for the source and microphone in the room.
    Returns:
        tuple: Random source and receiver positions (Sx, Sy, Sz, Rx, Ry, Rz).
    """
    position_range = np.arange(0.5, min(Lx, Ly, Lz) - 0.5, 0.5)
    Sx, Sy, Sz = [np.random.choice(position_range, size=1) for _ in range(3)]
    Rx, Ry, Rz = [np.random.choice(position_range, size=1) for _ in range(3)]
    while (Rx, Ry, Rz) == (Sx, Sy, Sz):
        Rx, Ry, Rz = [np.random.choice(position_range, size=1) for _ in range(3)]
    return Sx, Sy, Sz, Rx, Ry, Rz


def get_material():
    """
    Randomly choose materials for the room surfaces.
    Returns:
        dict: Material dictionary with random choices for walls, ceiling, and floor.
    """
    return {
        'east': np.random.choice(WALLS),
        'west': np.random.choice(WALLS),
        'north': np.random.choice(WALLS),
        'south': np.random.choice(WALLS),
        'ceiling': np.random.choice(CEIL),
        'floor': np.random.choice(FLOOR),
    }


def create_ambisonic_array(order_of_ambisonics, sample_rate):
    mic_radius = 0.000001
    order = order_of_ambisonics
    samples = (order + 1) ** 2
    mic_positions, orientations, degrees = HOA_array(samples=samples, radius=1, n_order=order)
    mic_positions = mic_positions * mic_radius
    microphone_directivities = []
    for i in range(samples):
        orientation = orientations[i]
        directivity = pra.directivities.SphericalHarmonicDirectivity(
            orientation, n=degrees[i][0], m=degrees[i][1]
        )
        microphone_directivities.append(directivity)
    return mic_positions.T, sample_rate, microphone_directivities


def random_point_in_polygon(polygon, height):
    """
    Generates a random point inside a given polygon using rejection sampling.

    Args:
        polygon (np.ndarray): 2D array of polygon vertices (shape: 2 x N).

    Returns:
        np.ndarray: A random point inside the polygon as [x, y].
    """
    poly = Polygon(polygon.T)
    min_x, min_y, max_x, max_y = poly.bounds
    min_z = 0
    max_z = height-0.5

    while True:
        # Generate random point within the bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_z, max_z)
        point = Point(x, y)
        if poly.contains(point):
            return np.array([x, y, z])


def scale_polygon_variable(vertices, scale_factors):
    """
    Scales a polygon variably, applying different scaling factors for x and y directions.

    Args:
        vertices (np.ndarray): The 2D vertices of the polygon (shape: 2 x N).
        scale_factors (dict): A dictionary with keys 'x' and 'y' for scaling along each axis.
    
    Returns:
        np.ndarray: Scaled vertices (2 x N).
    """
    scaled_vertices = vertices.copy()
    for i in range(vertices.shape[1]):
        scaled_vertices[0, i] *= scale_factors.get('x', 1.0)  # Scale x
        scaled_vertices[1, i] *= scale_factors.get('y', 1.0)  # Scale y
    return scaled_vertices

def makeCoupledPolygonRoom(fs, height=3.0, shape='L'):
    """
    Create a single complex polygon room, such as L-shaped or T-shaped.

    Parameters:
        height (float): Height of the room.

    Returns:
        room (pyroomacoustics.Room): The generated complex polygon room.
    """
    # Define an L-shaped room as a complex polygon
    # Adjust these points to create other shapes like T, etc.

    if shape=='L':
        corners = np.array([
                [0, 0],  # Bottom left
                [4, 0],  # Bottom right
                [4, 2],  # First inner corner (L shape start)
                [2, 2],  # Inner corner horizontal
                [2, 4],  # Vertical L shape
                [0, 4],  # Top left
            ]
        ).T
    elif shape=='T':
        corners = np.array([[0, 0], [6, 0], [6, 2], [4, 2], [4, 4], [2, 4], [2, 2], [0, 2]]).T

    elif shape == 'H':
        # H-shape polygon coordinates (2D)
        corners = np.array([[0,0], [0,5], [1,5], [1,2], [2,2], [2,5], [3,5], [3,0], [2,0], [2,1], [1,1], [1,0]]).T        
    else:
        shape ='C'
        corners = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [4, 3], [4, 4], [0, 4]]).T #C shape
    

    # Define variable scaling factors for x and y
    scale_factors = {
        'x': np.random.uniform(1, 5.0),  # Stretch/compress in x direction
        'y': np.random.uniform(1, 5.0)   # Stretch/compress in y direction
    }

    corners = scale_polygon_variable(corners, scale_factors)
    
    # Randomized wall absorption coefficients
    wall_absorptions =  np.random.choice(WALLS, (corners.shape[1],), replace=True)
    wall_materials = pra.make_materials(*[(mat,) for mat in wall_absorptions])

    # Extrude the polygon to 3D
    vertical_materials = pra.make_materials(floor=np.random.choice(FLOOR), ceiling=(np.random.choice(CEIL)))
    

    return wall_materials,vertical_materials, corners, height

def get_directive_EM32(mic_center, fs=16000):
    em = pra.MeasuredDirectivityFile("EM32_Directivity", fs=fs)
    positions = []
    directivities = []
    microphone_angles = [
        (0.0, 21.0), (32.0, 58.0), (0.0, 90.0), (-32.0, 58.0),
        (-45.0, 35.0), (-90.0, 0.0), (-135.0, 35.0), (-180.0, 21.0),
        (135.0, 35.0), (90.0, 0.0), (45.0, 35.0), (0.0, -21.0),
        (180.0, -21.0), (123.0, -58.0), (90.0, -90.0), (57.0, -58.0),
        (45.0, -35.0), (0.0, -159.0), (-45.0, -35.0), (-57.0, -58.0),
        (-90.0, -90.0), (-123.0, -58.0), (-135.0, -35.0), (180.0, -159.0),
        (135.0, -35.0), (123.0, -122.0), (90.0, -180.0), (57.0, -122.0),
        (45.0, -145.0), (0.0, 159.0), (-45.0, -145.0), (-57.0, -122.0)
    ]
    for i in range(32):
        positions.append(em.get_mic_position(f"EM_32_{i}"))
        rot = Rotation3D([microphone_angles[i][0], microphone_angles[i][1]], "yz", degrees=True)
        directivities.append(em.get_mic_directivity(f"EM_32_{i}", rot))
    
    return pra.MicrophoneArray((np.array(positions) + mic_center).T, fs, directivities)

def fibonacci_sphere(samples=25, radius=1.0):
    """
    Generates points on a sphere using the Fibonacci lattice method.
    This creates a near-uniform distribution of points on a sphere.
    
    Parameters:
    samples (int): Number of points to generate (for 4th order, we need 25).
    radius (float): Radius of the sphere.

    Returns:
    np.array: 3D coordinates of the points.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points.append([x * radius, y * radius, z * radius])

    return np.array(points)

def EM32_mic_config(num_mics=32, mic_center=[0,0,0], fs = 16000):

    all_positions = fibonacci_sphere(num_mics, radius=.042)
    all_positions = all_positions + mic_center
    
    return pra.MicrophoneArray(all_positions.T, fs)


def add_directive_microphones(room, position, rot_y, rot_z, fs, mic_type='kemar'):

    rot = Rotation3D([rot_y, rot_z], "yz", degrees=True)

    if mic_type == 'kemar':
        hrtf = MeasuredDirectivityFile(path="mit_kemar_normal_pinna.sofa", fs=fs)
        dir_left = hrtf.get_mic_directivity("left", orientation=rot)
        dir_right = hrtf.get_mic_directivity("right", orientation=rot)
        if room.n_mics == 0:
            room.add_microphone(position, directivity=dir_left)
            room.add_microphone(position, directivity=dir_right)
        else:
            room.mic_array = None
            room.add_microphone(position, directivity=dir_left)
            room.add_microphone(position, directivity=dir_right)            

    elif mic_type == 'em_32':
        eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=fs)
        dir_obj = eigenmike.get_mic_directivity(f"EM_32_9", orientation=rot)
        if room.n_mics == 0:
            room.add_microphone(position, directivity=dir_obj)
        else:
            room.mic_array = None
            room.add_microphone(position, directivity=dir_obj)
    else:
        hrtf = MeasuredDirectivityFile(path="mit_kemar_normal_pinna.sofa", fs=fs)
        dir_left = hrtf.get_mic_directivity("left", orientation=rot)
        dir_right = hrtf.get_mic_directivity("right", orientation=rot)
        room.add_microphone(position, directivity=dir_left)
        room.add_microphone(position, directivity=dir_right)

        eigenmike = get_directive_EM32(position, fs)
        room.add(eigenmike)


    return room

def add_directive_source(room, position, rot_y, rot_z, fs):
    '''
    speaker_type
        - Genelec_8020
        - Lambda_labs_CX-1A
        - HATS_4128C
        - Tannoy_System_1200
        - Neumann_KH120A
        - Yamaha_DXR8
        - BM_1x12inch_driver_closed_cabinet
        - BM_1x12inch_driver_open_cabinet
        - BM_open_stacked_on_closed_withCrossoverNetwork
        - BM_open_stacked_on_closed_fullrange
        - Palmer_1x12inch
        - Vibrolux_2x10inch
    '''
    rot = Rotation3D([rot_y, rot_z], "yz", degrees=True)
    measurements = MeasuredDirectivityFile("LSPs_HATS_GuitarCabinets_Akustikmessplatz", fs=fs)
    for cur_speaker in source_types:
        source_directivity = measurements.get_source_directivity(cur_speaker,rot)
        room.add_source(position, source_directivity)    
    
    '''if len(room.sources) == 0 :
        room.add_source(position, source_directivity)
    else:
        room.sources = []
        room.add_source(position, source_directivity)'''
    return room

def get_directive_microphones(rot_y, rot_z, fs, mic_type='kemar'):

    rot = Rotation3D([rot_y, rot_z], "yz", degrees=True)

    if mic_type == 'kemar':
        hrtf = MeasuredDirectivityFile(path="mit_kemar_normal_pinna.sofa", fs=fs)
        dir_left = hrtf.get_mic_directivity("left", orientation=rot)
        dir_right = hrtf.get_mic_directivity("right", orientation=rot)
        return dir_left, dir_right

    elif mic_type == 'em_32':
        eigenmike = MeasuredDirectivityFile("EM32_Directivity", fs=fs)
        dir_obj = eigenmike.get_mic_directivity("EM_32_9", orientation=rot)
        return dir_obj

def get_directive_source(room, position, rot_y, rot_z, fs, speaker_type):
    '''
    speaker_type
        - Genelec_8020
        - Lambda_labs_CX-1A
        - HATS_4128C
        - Tannoy_System_1200
        - Neumann_KH120A
        - Yamaha_DXR8
        - BM_1x12inch_driver_closed_cabinet
        - BM_1x12inch_driver_open_cabinet
        - BM_open_stacked_on_closed_withCrossoverNetwork
        - BM_open_stacked_on_closed_fullrange
        - Palmer_1x12inch
        - Vibrolux_2x10inch
    '''
    rot = Rotation3D([rot_y, rot_z], "yz", degrees=True)
    measurements = MeasuredDirectivityFile("LSPs_HATS_GuitarCabinets_Akustikmessplatz", fs=fs)
    source_directivity = measurements.get_source_directivity(speaker_type,rot)
    room.add_source(position, source_directivity)
    return room

mic_types = ['em_32'] #'kemar',
source_types = ['point'] #'Genelec_8020', 'HATS_4128C', 'Lambda_labs_CX-1A'

def shoebox_to_vertices(x, y, z):
    """
    Converts shoebox dimensions (x, y, z) into 3D vertices representing the corners of the room.
    
    Parameters:
    - x (float): Length of the room (along x-axis)
    - y (float): Width of the room (along y-axis)
    - z (float): Height of the room (along z-axis)
    
    Returns:
    - vertices (numpy.ndarray): A (8, 3) array where each row represents a vertex (x, y, z).
    """
    # Define the 8 vertices of a rectangular cuboid (shoebox)
    vertices = np.array([
        [0, 0, 0],  # vertex 1
        [x, 0, 0],  # vertex 2
        [x, y, 0],  # vertex 3
        [0, y, 0],  # vertex 4
        [0, 0, z],  # vertex 5
        [x, 0, z],  # vertex 6
        [x, y, z],  # vertex 7
        [0, y, z],  # vertex 8
    ])
    
    return vertices
    
def polygon_to_3d_vertices(x_coords, y_coords, z_height):
    """
    Converts 2D polygon (x, y) coordinates and height into 3D vertices.
    
    Parameters:
    - x_coords (list): List of x-coordinates of the polygon vertices.
    - y_coords (list): List of y-coordinates of the polygon vertices.
    - z_height (float): Height of the room (z-coordinate).
    
    Returns:
    - vertices (numpy.ndarray): A (2 * len(x_coords), 3) array where each row represents a vertex (x, y, z).
    """
    # Number of vertices in the polygon
    num_vertices = len(x_coords)

    # Create the floor vertices (z=0)
    floor_vertices = np.array(list(zip(x_coords, y_coords, np.zeros(num_vertices))))
    
    # Create the ceiling vertices (z=z_height)
    ceiling_vertices = np.array(list(zip(x_coords, y_coords, np.full(num_vertices, z_height))))
    
    # Combine floor and ceiling vertices
    vertices = np.vstack([floor_vertices, ceiling_vertices])
    
    return vertices

def save_rirs(output_path, name, fs, room):
    for j, cur_source_type in enumerate(source_types):
        #binaural_rir = np.array([RIR[0][j], RIR[1][j]])
        RIR = room.rir
        max_len = np.array([len(RIR[i][0]) for i in range(room.n_mics)]).max()
        em_rir = np.array([np.pad(RIR[i][j], (0,max_len - RIR[i][j].shape[0])).tolist() for i in range(room.n_mics)])
        em_rir = em_rir[...,:fs*(max_len//fs)]
        os.makedirs(f'{output_path}', exist_ok=True)
        os.makedirs(f'{output_path}/valid/', exist_ok=True)
        #os.makedirs(f'{output_path}/{mic_types[1]}/{cur_source_type}', exist_ok=True)
        #wavfile.write(f"{output_path}/{mic_types[0]}/{cur_source_type}/{name}", fs, binaural_rir.astype(np.float32).T)
        #wavfile.write(f"{output_path}/{name}", fs, em_rir.astype(np.float32).T)
        wavfile.write(f"{output_path}/valid/{name}", fs, RIR[0][0].astype(np.float32).T)
            

def generate_rir_for_all_combinations_together(room, output_path, mic_position, source_position, i, j, fs):   
    #randomize orientations later
    or_y = 0
    or_z = 0
    #room = add_directive_microphones(room, mic_position, or_y, or_z, fs, 'all')
    #room = add_directive_source(room, source_position, or_y, or_z, fs)
    mics = EM32_mic_config(2, mic_position, fs)
    room.add(mics)
    room.add_source(source_position)
    room.image_source_model()
    room.set_ray_tracing(10000, 0.25, energy_thres=1e-10, time_thres=5)
    room.ray_tracing()
    room.compute_rir()

    name = f'R{i}_P{j}_IR.wav'
    RT = pra.experimental.rt60.measure_rt60(room.rir[0][0], fs=fs, decay_db=60)
    save_rirs(output_path, name, fs, room)
    
    return RT, or_y, or_z

'''def generate_mic_source_pair_rir(room, output_path, mic_position, source_position, i, j, fs):   
    #randomize orientations later
    or_y = 0
    or_z = 0
    for cur_mic_type in mic_types:
        room = add_directive_microphones(room, mic_position, or_y, or_z, fs, cur_mic_type)
        for cur_source_type in source_types:  
            room = add_directive_source(room, source_position, or_y, or_z, fs, cur_source_type)
            room.image_source_model()
            room.compute_rir()
            RIR = np.stack([np.pad(r[0], (0, fs - r[0].shape[0])) for r in room.rir], axis=0)
            name = f'R{i}_P{j}_IR.wav'
            os.makedirs(f'{output_path}', exist_ok=True)            
            wavfile.write(f"{output_path}/{name}", fs, RIR.astype(np.float32).T)
            RT = pra.experimental.rt60.measure_rt60(RIR[0], fs=room.fs, decay_db=60)
            room.sources.clear()
    return RT, or_y, or_z'''

def simulate_room(i, j, output_path, fs, wall_materials, vertical_materials, corners, height, regularity):           
    # Create the 2D polygonal room
    room = pra.Room.from_corners(
        corners,
        fs=fs,
        t0=0.0,
        max_order=2,
        materials=wall_materials, ray_tracing=True
    )
    room.extrude(height, materials=vertical_materials) 
    # Add a source randomly within the room
    x_source, y_source, z_source = random_point_in_polygon(corners, height)
    x_mic, y_mic, z_mic = random_point_in_polygon(corners, height)
    while int(x_source) == int(x_mic) and int(y_source) == int(y_mic) and int(z_source) == int(z_mic):
        x_source, y_source, z_source = random_point_in_polygon(corners, height)
        x_mic, y_mic, z_mic = random_point_in_polygon(corners, height)
    
    RT, or_y, or_z = generate_rir_for_all_combinations_together(room, f'{output_path}/', [x_mic, y_mic, z_mic], [x_source, y_source, z_source], i, j, fs)
    vertices = polygon_to_3d_vertices(corners[0, :].tolist(), corners[1, :].tolist(), height)
    return {
    "Name": f"R{i}_P{j}_IR.wav",
    "RT": RT,
    "vertices": vertices.tolist(),
    "x_source": x_source,
    "y_source": y_source,
    "z_source": z_source,
    "x_mic": x_mic,
    "y_mic": y_mic,
    "z_mic": z_mic,
    "room_type": regularity}

def simulate_shoebox_room(i, j, output_path, fs, mat, x_room, y_room, z_room, regularity='shoebox'):
    room = pra.ShoeBox(
                    p=[x_room, y_room, z_room],
                    materials=pra.make_materials(**mat),
                    fs=fs,
                    max_order=3, ray_tracing=True)
    x_source, y_source, z_source, x_mic, y_mic, z_mic = get_random_positions(x_room, y_room, z_room)
    mic_center = [x_mic.item(), y_mic.item(), z_mic.item()]
    #room.set_ray_tracing(7000, 0.1, energy_thres=1e-8, time_thres=5)
    RT, or_y, or_z = generate_rir_for_all_combinations_together(room, f'{output_path}', mic_center, [x_source, y_source, z_source], i, j, fs)
    vertices = shoebox_to_vertices(x_room.item(), y_room.item(), z_room.item())    
    return {
    "Name": f"R{i}_P{j}_IR.wav",
    "RT": RT,
    "vertices": vertices.tolist(),
    "x_source": x_source.item(),
    "y_source": y_source.item(),
    "z_source": z_source.item(),
    "x_mic": x_mic.item(),
    "y_mic": y_mic.item(),
    "z_mic": z_mic.item(),
    "room_type": regularity
    #"colatitude_deg": or_y,
    #"azimuth_deg": or_z
    }

def main(output_path, num_rooms, num_positions):
    data = []
    fs = 48000
    #ambisonic_microphone = create_ambisonic_array(order_of_ambisonics=7, sample_rate=48000)
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_rooms):
            regularity = np.random.choice(np.array(['shoebox', 'L', 'T', 'H', 'C']), p= [.3, .2, .2, .1, .2]) #[.3, .2, .2, .1, .2]
            
            if regularity == 'shoebox':
                x_room, y_room, z_room = get_random_dimensions()
                mat = get_material()             
                for j in range(num_positions):                                
                    futures.append(executor.submit(simulate_shoebox_room, i, j, output_path, fs, mat, x_room, y_room, z_room))
            else:
                height = np.random.choice(np.arange(2, 8, 0.5))
                wall_mat, vert_mat, corners, height = makeCoupledPolygonRoom(fs, height=height, shape=regularity)
                for j in range(num_positions):                                
                    futures.append(executor.submit(simulate_room, i, j, output_path, fs, wall_mat, vert_mat, corners, height, regularity))
        for future in tqdm(futures):
            data.append(future.result())
    pd.DataFrame(data).to_csv(f"{output_path}/valid/Generated_HOA_SRIR_data.csv", index=False)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main("./datasetV2", num_rooms=100, num_positions=10) #500,20
