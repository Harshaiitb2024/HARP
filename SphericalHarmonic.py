import pyroomacoustics as pra
from scipy.special import sph_harm
import numpy as np

def compute_spherical_harmonics(order, theta, phi):
    """Computes spherical harmonics up to a given order.

    Args:
        order (int): The maximum order of spherical harmonics to compute.
        theta (float or numpy.ndarray): The polar angle(s) (colatitude) in radians.
        phi (float or numpy.ndarray): The azimuthal angle(s) in radians.

    Returns:
        numpy.ndarray: An array of complex spherical harmonic values.
                       The shape of the output depends on the input shape of theta and phi.
    """

    harmonics = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            harmonics.append(sph_harm(m, n, phi, theta))
    return np.array(harmonics)

def generate_hoa_array(num_microphones=25, radius=1.0, ambisonic_order=4):
    """Generates microphone positions and orientations for an HOA array.

    Uses a Fibonacci sphere algorithm for uniform distribution of microphones.

    Args:
        num_microphones (int): The number of microphones in the array.
        radius (float): The radius of the sphere on which microphones are placed.
        ambisonic_order (int): The order of Ambisonics.

    Returns:
        tuple: A tuple containing:
            - positions (numpy.ndarray): Array of 3D microphone positions (shape: (num_microphones, 3)).
            - orientations (list): List of DirectionVector objects for each microphone.
            - degrees (list): List of (n, m) degree tuples for each microphone.

    Raises:
        ValueError: If num_microphones is less than (ambisonic_order+1)^2
    """
    if num_microphones < (ambisonic_order + 1)**2:
        raise ValueError(f"Number of microphones ({num_microphones}) is insufficient for Ambisonic order {ambisonic_order}. Needs at least {(ambisonic_order + 1)**2}")

    positions = np.zeros((num_microphones, 3))
    orientations = []
    degrees = []

    offset = 2.0 / num_microphones
    increment = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(num_microphones):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y**2)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        positions[i] = np.array([x, y, z]) * radius

        azimuth = np.arctan2(y, x)
        colatitude = np.arccos(z)
        orientation = pra.directivities.DirectionVector(azimuth, colatitude, degrees=False)
        orientations.append(orientation)
    
    # Assign spherical harmonic degrees (n, m) to each microphone
    degree_index = 0
    for n in range(ambisonic_order + 1):  # n goes from 0 to n_order
        for m in range(-n, n + 1):  # m goes from -n to +n
            if degree_index < num_microphones:
                degrees.append((n, m))
                degree_index += 1
            else:
                break
    return positions, orientations, degrees

from pyroomacoustics import Directivity, all_combinations
import matplotlib.pyplot as plt

class SphericalHarmonicDirectivity(Directivity):
    """Directivity based on spherical harmonics.

    Parameters
    ----------
    orientation : DirectionVector
        Indicates the direction of the pattern.
    n : int
        The order of the spherical harmonic.
    m : int
        The degree of the spherical harmonic.
    """

    def __init__(self, orientation, n, m):
        super().__init__(orientation)  # Use super() for inheritance
        self._n = n  # Spherical harmonic order
        self._m = m  # Spherical harmonic degree
        

    def sn3d_normalization(self):
        """Computes the SN3D normalization factor.

        Returns:
            float: SN3D normalization factor.
        """
        return 1 / np.sqrt(2 * self._n + 1)

    def complex_to_real_sph_harm(self, m, n, azimuth, colatitude):
        """Converts complex spherical harmonics to real spherical harmonics.

        Args:
            m (int): Order of the spherical harmonic.
            n (int): Degree of the spherical harmonic.
            azimuth (float or array_like): Azimuth angle(s) in radians.
            colatitude (float or array_like): Colatitude angle(s) in radians.

        Returns:
            float or array_like: Real spherical harmonics value(s).
        """
        Y_complex = sph_harm(m, n, azimuth, colatitude)

        if m > 0:
            Y_real = np.sqrt(2) * (-1)**m * np.real(Y_complex)
        elif m < 0:
            Y_real = np.sqrt(2) * (-1)**abs(m) * np.imag(Y_complex)
        else:  # m == 0
            Y_real = np.real(Y_complex)
        return Y_real

    def get_response(self, azimuth, colatitude=None, magnitude=False, degrees=True):
        """Gets the response for provided angles using spherical harmonics.

        Args:
            azimuth (array_like): Azimuth in degrees or radians.
            colatitude (array_like, optional): Colatitude in degrees or radians.
            magnitude (bool, optional): Whether to return the magnitude of the response.
            degrees (bool, optional): Whether provided angles are in degrees.

        Returns:
            numpy.ndarray: Spherical harmonic response at provided angles.

        Raises:
            AssertionError: If input angles are invalid.
        """
        if colatitude is not None:
            assert len(azimuth) == len(colatitude), "Azimuth and colatitude must have the same length."

        if degrees:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)

        # Default to elevation if colatitude is not provided
        if colatitude is None:
            colatitude = np.pi / 2 - azimuth

        colatitude = np.clip(colatitude, a_min=0, a_max=np.pi)
        azimuth = azimuth % (2 * np.pi) # ensures azimuth is within [0, 2pi]

        assert colatitude.max() <= np.pi and colatitude.min() >= 0, "Colatitude must be within [0, pi]"
        assert azimuth.max() <= 2 * np.pi and azimuth.min() >= 0, "Azimuth must be within [0, 2pi]"

        resp = self.complex_to_real_sph_harm(self._m, self._n, azimuth, colatitude) * 0.5 #* self.sn3d_normalization()

        return np.abs(resp) if magnitude else resp

    def plot_response(self, azimuth, colatitude=None, degrees=True, ax=None, offset=None, axis_at_zero=False):
        """Plots the spherical harmonic directivity response.

        Args:
            azimuth (array_like): Azimuth values for plotting.
            colatitude (array_like, optional): Colatitude values for plotting.
            degrees (bool): Whether provided values are in degrees.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on.
            offset (list, optional): 3D coordinates of the plot offset.
            axis_at_zero (bool): if True, plots axis at origin

        Returns:
            matplotlib.axes.Axes: The axes object with the plot.
        """
        if offset is None:
            offset = [0, 0, 0]
        x_offset, y_offset, z_offset = offset

        if degrees:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)

        if colatitude is not None:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

            spher_coord = all_combinations(azimuth, colatitude)
            azi_flat = spher_coord[:, 0]
            col_flat = spher_coord[:, 1]
            resp = self.get_response(azimuth=azi_flat, colatitude=col_flat, magnitude=True, degrees=False)
            RESP = resp.reshape(len(azimuth), len(colatitude))

            AZI, COL = np.meshgrid(azimuth, colatitude)
            X = RESP.T * np.sin(COL) * np.cos(AZI) + x_offset
            Y = RESP.T * np.sin(COL) * np.sin(AZI) + y_offset
            Z = RESP.T * np.cos(COL) + z_offset

            ax.plot_surface(X, Y, Z)

            if axis_at_zero:
                for v in range(3):
                    val = [1,0,0]
                    x = [val[v-0], -val[v-0]]
                    y = [val[v-1], -val[v-1]]
                    z = [val[v-2], -val[v-2]]
                    ax.plot(x,y,z,'k-', linewidth=0.25)
                ax.axis('off')
                ax.grid(b=None)

            ax.set_title(f"Spherical Harmonic Order {self._n}, Degree {self._m}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            if ax is None:
                fig, ax = plt.subplots()

            resp = self.get_response(azimuth=azimuth, magnitude=True, degrees=False)
            X = resp * np.cos(azimuth) + x_offset
            Y = resp * np.sin(azimuth) + y_offset
            ax.plot(X, Y)
            ax.set_title(f"Spherical Harmonic Order {self._n}, Degree {self._m}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        return ax
