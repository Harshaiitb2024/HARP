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
    """Return random room dimensions as floats (not arrays)."""
    dimensions_range = np.arange(min_dim, max_dim, step)
    return tuple(float(np.random.choice(dimensions_range)) for _ in range(3))

def get_random_positions(Lx: float, Ly: float, Lz: float,
                         min_dist_from_wall: float = 0.5, step: float = 0.5) -> tuple:
    """Return random source and mic positions as floats (not arrays)."""
    x_pos_range = np.arange(min_dist_from_wall, Lx - min_dist_from_wall, step)
    y_pos_range = np.arange(min_dist_from_wall, Ly - min_dist_from_wall, step)
    z_pos_range = np.arange(min_dist_from_wall, Lz - min_dist_from_wall, step)

    if len(x_pos_range) == 0 or len(y_pos_range) == 0 or len(z_pos_range) == 0:
        raise ValueError(f"Room dimensions ({Lx}, {Ly}, {Lz}) too small.")

    S1x, S1y, S1z = [float(np.random.choice(r)) for r in [x_pos_range, y_pos_range, z_pos_range]]
    while True:
        S2x, S2y, S2z = [float(np.random.choice(r)) for r in [x_pos_range, y_pos_range, z_pos_range]]
        if np.linalg.norm([S1x - S2x, S1y - S2y, S1z - S2z]) > 0.5:
            break
    Rx, Ry, Rz = [float(np.random.choice(r)) for r in [x_pos_range, y_pos_range, z_pos_range]]

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
        room_dims = [Lx, Ly, Lz]
        source1_pos = [S1x, S1y, S1z]
        source2_pos = [S2x, S2y, S2z]
        mic_center_pos = [Rx, Ry, Rz]

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

        room.image_source_model()
        room.compute_rir()

        # --- RT60-based trimming ---
        try:
            rt60 = pra.experimental.measure_rt60(room.rir, room.fs)
            rt60_max = np.nanmax(rt60)
            if np.isnan(rt60_max) or rt60_max <= 0:
                raise ValueError("Invalid RT60")
        except Exception:
            logging.warning(f"RT60 estimation failed for {room_name}, using fallback=3s")
            rt60_max = 3.0

        target_len = int(np.ceil(rt60_max * room.fs * 1.1))

        num_mics = len(room.mic_array.R[0])
        num_sources = len(room.sources)
        RIRs = np.zeros((num_sources, num_mics, target_len), dtype=np.float32)

        # Pad/trim each rir to target_len
        for src_idx in range(num_sources):
            for mic_idx in range(num_mics):
                rir = np.asarray(room.rir[mic_idx][src_idx], dtype=np.float32)
                if rir.size == 0:
                    rir = np.zeros(target_len, dtype=np.float32)
                elif len(rir) < target_len:
                    rir = np.pad(rir, (0, target_len - len(rir)))
                else:
                    rir = rir[:target_len]
                RIRs[src_idx, mic_idx, :] = rir

        # Normalize to [-1, 1]
        max_rir_val = np.max(np.abs(RIRs))
        if max_rir_val > 0:
            RIRs = RIRs / max_rir_val

        # Save compressed
        np.savez_compressed(output_filepath, rirs=RIRs, fs=room.fs)

        return {
            "Name": npz_filename,
            "x_room": Lx, "y_room": Ly, "z_room": Lz,
            "x_source1": S1x, "y_source1": S1y, "z_source1": S1z,
            "x_source2": S2x, "y_source2": S2y, "z_source2": S2z,
            "x_mic": Rx, "y_mic": Ry, "z_mic": Rz,
            "rt60": rt60_max,
            "status": "Success"
        }

    except Exception as e:
        return {
            "Name": npz_filename,
            "status": f"Error: {str(e)}",
            "x_room": Lx, "y_room": Ly, "z_room": Lz
        }

def main(output_path: str, num_rooms: int, num_positions: int, ambi_order: int = 3, sample_rate: int = 16000):
    os.makedirs(output_path, exist_ok=True)
    ambisonic_microphone = create_ambisonic_array(order_of_ambisonics=ambi_order, sample_rate=sample_rate)

    all_results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(num_rooms), desc="Generating Rooms"):
            Lx, Ly, Lz = get_random_dimensions()
            mat = get_random_materials()
            for j in range(num_positions):
                S1x, S1y, S1z, S2x, S2y, S2z, Rx, Ry, Rz = get_random_positions(Lx, Ly, Lz)
                physical_params = [Lx, Ly, Lz, mat, S1x, S1y, S1z, S2x, S2y, S2z, Rx, Ry, Rz]
                futures.append(executor.submit(simulate_room, i, j, output_path, ambisonic_microphone, physical_params))

        for future in tqdm(futures, desc="Processing Results"):
            result = future.result()
            all_results.append(result)

    data_df = pd.DataFrame([r for r in all_results if r])
    data_output_path = os.path.join(output_path, "Generated_HOA_SRIR_data.csv")
    data_df.to_csv(data_output_path, index=False)
    logging.info(f"Saved metadata for {len(data_df)} RIRs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2-source RIRs and save as NPZ.")
    parser.add_argument("--output_path", type=str, default="./Generated_HOA_IRs/")
    parser.add_argument("--num_rooms", type=int, default=2)
    parser.add_argument("--num_positions", type=int, default=10)
    parser.add_argument("--ambi_order", type=int, default=3)
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    main(
        output_path=args.output_path,
        num_rooms=args.num_rooms,
        num_positions=args.num_positions,
        ambi_order=args.ambi_order,
        sample_rate=args.sample_rate
    )
