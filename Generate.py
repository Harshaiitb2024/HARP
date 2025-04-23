import numpy as np
import pandas as pd
from scipy.io import wavfile
import pyroomacoustics as pra
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from SphericalHarmonic import HOA_array

# Global Constants
LOOKUP_TABLE = [
    "hard_surface",
    "brickwork",
    "rough_concrete",
    "unpainted_concrete",
    "rough_lime_wash",
    "smooth_brickwork_flush_pointing",
    "smooth_brickwork_10mm_pointing",
    "brick_wall_rough",
    "ceramic_tiles",
    "limestone_wall",
    "reverb_chamber",
    "plasterboard",
    "wooden_lining",
    "glass_3mm",
    "glass_window",
    "double_glazing_30mm",
    "double_glazing_10mm",
    "wood_1.6cm",
    "curtains_cotton_0.5",
    "curtains_0.2",
    "curtains_velvet",
    "curtains_glass_mat",
    "carpet_cotton",
    "carpet_6mm_closed_cell_foam",
    "carpet_6mm_open_cell_foam",
    "carpet_tufted_9m",
    "felt_5mm",
    "carpet_hairy",
    "concrete_floor",
    "marble_floor",
    "orchestra_1.5_m2",
    "panel_fabric_covered_6pcf",
    "panel_fabric_covered_8pcf",
    "ceiling_fibre_abosrber",
]
WALLS = np.concatenate((LOOKUP_TABLE[:22], LOOKUP_TABLE[30:]))
FLOOR = np.concatenate((LOOKUP_TABLE[7:8], LOOKUP_TABLE[22:31]))
CEIL = np.concatenate((LOOKUP_TABLE[3:5], LOOKUP_TABLE[30:]))


# Utility Functions
def get_random_dimensions():
    """
    Generate random dimensions for the room.
    Returns:
        tuple: Random dimensions (Lx, Ly, Lz).
    """
    dimensions_range = np.arange(4, 15, 1)
    return tuple(np.random.choice(dimensions_range, size=1) for _ in range(3))


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
        "east": np.random.choice(WALLS),
        "west": np.random.choice(WALLS),
        "north": np.random.choice(WALLS),
        "south": np.random.choice(WALLS),
        "ceiling": np.random.choice(CEIL),
        "floor": np.random.choice(FLOOR),
    }

def create_ambisonic_array(order_of_ambisonics, sample_rate):
    mic_radius = 0.000001
    order = order_of_ambisonics
    samples = (order + 1) ** 2
    mic_positions, orientations, degrees = HOA_array(
        samples=samples, radius=1, n_order=order
    )
    mic_positions = mic_positions * mic_radius
    microphone_directivities = []
    for i in range(samples):
        orientation = orientations[i]
        directivity = pra.directivities.SphericalHarmonicDirectivity(
            orientation, n=degrees[i][0], m=degrees[i][1]
        )
        microphone_directivities.append(directivity)
    return mic_positions.T, sample_rate, microphone_directivities


def simulate_room(i, j, output_path, ambisonic_microphone, physical_params):
    x_room, y_room, z_room, mat, x_source, y_source, z_source, x_mic, y_mic, z_mic = (
        physical_params
    )
    mic_center = np.array([[x_mic.item(), y_mic.item(), z_mic.item()]])
    room = pra.ShoeBox(
        p=[x_room.item(), y_room.item(), z_room.item()],
        materials=pra.make_materials(**mat),
        fs=48000,
        max_order=30,
    )
    room.add_source([x_source.item(), y_source.item(), z_source.item()])
    room.add_microphone_array(
        pra.MicrophoneArray(
            ambisonic_microphone[0] + mic_center.T,
            ambisonic_microphone[1],
            ambisonic_microphone[2],
        )
    )
    room.image_source_model()
    room.compute_rir()

    max_len = np.array([len(r[0]) for r in room.rir]).max()
    seconds = np.ceil(max_len / 48000)

    RIR = np.stack(
        [np.pad(r[0], (0, int(seconds * 48000) - r[0].shape[0])) for r in room.rir],
        axis=0,
    )
    RIR = RIR / RIR.max()

    wavfile.write(f"{output_path}/R{i}_P{j}_IR.wav", 48000, RIR.astype(np.float32).T)
    RT = pra.experimental.rt60.measure_rt60(RIR[0], fs=room.fs, decay_db=30)
    c50 = 10 * np.log10(
        sum(np.square(np.abs(RIR[0, : int(room.fs * 0.05)]))) / sum(np.square(np.abs(RIR[0, int(room.fs * 0.05):])))
    )
    return {
        "Name": f"R{i}_P{j}_IR.wav",
        "RT": RT,
        "c50": c50,
        "x_room": x_room[0],
        "y_room": y_room[0],
        "z_room": z_room[0],
        "x_source": x_source[0],
        "y_source": y_source[0],
        "z_source": z_source[0],
        "x_mic": x_mic[0],
        "y_mic": y_mic[0],
        "z_mic": z_mic[0],
    }


def main(output_path, num_rooms, num_positions):
    data = []
    ambisonic_microphone = create_ambisonic_array(
        order_of_ambisonics=7, sample_rate=48000
    )
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(num_rooms)):
            x_room, y_room, z_room = get_random_dimensions()
            mat = get_material()
            for j in range(num_positions):
                x_source, y_source, z_source, x_mic, y_mic, z_mic = (
                    get_random_positions(x_room, y_room, z_room)
                )
                physical_params = [
                    x_room,
                    y_room,
                    z_room,
                    mat,
                    x_source,
                    y_source,
                    z_source,
                    x_mic,
                    y_mic,
                    z_mic,
                ]
                futures.append(
                    executor.submit(
                        simulate_room,
                        i,
                        j,
                        output_path,
                        ambisonic_microphone,
                        physical_params,
                    )
                )
        for future in tqdm(futures):
            data.append(future.result())
    pd.DataFrame(data).to_csv(f"{output_path}/Generated_HOA_SRIR_data.csv", index=False)


if __name__ == "__main__":
    main("./Generated_HOA_IRs/", num_rooms=10000, num_positions=10)  # 500,20
