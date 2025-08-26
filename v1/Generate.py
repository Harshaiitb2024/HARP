import numpy as np
import pandas as pd
import pyroomacoustics as pra
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from SphericalHarmonic import HOA_array, SphericalHarmonicDirectivity
import os
import logging
import argparse
from typing import Union

# Configure logging
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, "rir_generation.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path)]
)

# --- Global Constants ---
LOOKUP_TABLE = [
    "hard_surface", "brickwork", "rough_concrete", "unpainted_concrete",
    "rough_lime_wash", "smooth_brickwork_flush_pointing", "smooth_brickwork_10mm_pointing",
    "brick_wall_rough", "ceramic_tiles", "limestone_wall", "reverb_chamber", "plasterboard",
    "wooden_lining", "glass_3mm", "glass_window", "double_glazing_30mm", "double_glazing_10mm",
    "wood_1.6cm", "curtains_cotton_0.5", "curtains_0.2", "curtains_velvet", "curtains_glass_mat",
    "carpet_cotton", "carpet_6mm_closed_cell_foam", "carpet_6mm_open_cell_foam", "carpet_tufted_9m",
    "felt_5mm", "carpet_hairy", "concrete_floor", "marble_floor", "orchestra_1.5_m2",
    "panel_fabric_covered_6pcf", "panel_fabric_covered_8pcf", "ceiling_fibre_abosrber"
]

WALLS = np.concatenate((LOOKUP_TABLE[:22], LOOKUP_TABLE[30:]))
FLOOR = np.concatenate((LOOKUP_TABLE[7:8], LOOKUP_TABLE[22:31]))
CEIL = np.concatenate((LOOKUP_TABLE[3:5], LOOKUP_TABLE[30:]))

# --- Utility Functions ---
def get_random_dimensions(min_dim: int = 4, max_dim: int = 15, step: int = 1) -> tuple:
    dimensions_range = np.arange(min_dim, max_dim, step)
    return tuple(np.random.choice(dimensions_range, size=1) for _ in range(3))

def get_random_positions(Lx: float, Ly: float, Lz: float, min_dist_from_wall: float = 0.5, step: float = 0.5) -> tuple:
    x_pos_range = np.arange(min_dist_from_wall, Lx - min_dist_from_wall, step)
    y_pos_range = np.arange(min_dist_from_wall, Ly - min_dist_from_wall, step)
    z_pos_range = np.arange(min_dist_from_wall, Lz - min_dist_from_wall, step)
    if len(x_pos_range) == 0 or len(y_pos_range) == 0 or len(z_pos_range) == 0:
        raise ValueError(f"Room dimensions ({Lx}, {Ly}, {Lz}) too small.")

    S1x, S1y, S1z = [np.random.choice(r, size=1) for r in [x_pos_range, y_pos_range, z_pos_range]]
    while True:
        S2x, S2y, S2z = [np.random.choice(r, size=1) for r in [x_pos_range, y_pos_range, z_pos_range]]
        if np.linalg.norm([S1x - S2x, S1y - S2y, S1z - S2z]) > 0.5:
            break
    Rx, Ry, Rz = [np.random.choice(r, size=1) for r in [x_pos_range, y_pos_range, z_pos_range]]
    return S1x, S1y, S1z, S2x, S2y, S2z, Rx, Ry, Rz

def get_random_materials() -> dict:
    return {
        "east": np.random.choice(WALLS),
        "west": np.random.choice(WALLS),
        "north": np.random.choice(WALLS),
        "south": np.random.choice(WALLS),
        "ceiling": np.random.choice(CEIL),
        "floor": np.random.choice(FLOOR),
    }

def create_ambisonic_array(order_of_ambisonics: int, sample_rate: int):
    mic_radius = 0.000001
    order = order_of_ambisonics
    samples = (order + 1) ** 2
    mic_positions, orientations, degrees = HOA_array(samples=samples, radius=1, n_order=order)
    mic_positions = mic_positions * mic_radius
    microphone_directivities = [
        SphericalHarmonicDirectivity(orientations[i], n=degrees[i][0], m=degrees[i][1])
        for i in range(samples)
    ]
    return mic_positions.T, sample_rate, microphone_directivities

def simulate_room(room_idx: int, pos_idx: int, output_path: str, ambisonic_microphone, physical_params: list) -> Union[dict, None]:
    room_name = f"R{room_idx}_P{pos_idx}"
    npz_filename = f"{room_name}_IR.npz"
    output_filepath = os.path.join(output_path, npz_filename)

    Lx, Ly, Lz, materials, S1x, S1y, S1z, S2x, S2y, S2z, Rx, Ry, Rz = physical_params

    try:
        room_dims = [Lx[0], Ly[0], Lz[0]]
        source1_pos = [S1x[0], S1y[0], S1z[0]]
        source2_pos = [S2x[0], S2y[0], S2z[0]]
        mic_center_pos = [Rx[0], Ry[0], Rz[0]]

        # Build room
        room = pra.ShoeBox(
            p=room_dims,
            materials=pra.make_materials(**materials),
            fs=ambisonic_microphone[1],
            max_order=30,
        )
        room.add_source(source1_pos)
        room.add_source(source2_pos)
        room.add_microphone_array(
            pra.MicrophoneArray(
                ambisonic_microphone[0] + np.array(mic_center_pos)[:, None],
                ambisonic_microphone[1],
                ambisonic_microphone[2],
            )
        )

        room.image_s
