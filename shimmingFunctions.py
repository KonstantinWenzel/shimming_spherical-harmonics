import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import isfile, exists
from scipy.constants import mu_0
# from numba import njit


def calcDipolMomentAnalytical(remanence, volume):
    """ Calculating the magnetic moment from the remanence in T and the volume in m^3"""
    m = remanence * volume / mu_0  # [A * m^2]
    return m


def plotSimple(data, FOV, fig, ax, cbar=True, **args):
    """ Generate simple colorcoded plot of 2D grid data with contour. Returns axes object."""
    im = ax.imshow(data, extent=FOV, origin="lower", **args)
    cs = ax.contour(data, colors="k", extent=FOV, origin="lower", linestyles="dotted")

    class nf(float):
        def __repr__(self):
            s = f"{self:.1f}"
            return f"{self:.0f}" if s[-1] == "0" else s

    cs.levels = [nf(val) for val in cs.levels]
    if plt.rcParams["text.usetex"]:
        fmt = r"%r"
    else:
        fmt = "%r"
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)
    if cbar == True:
        fig.colorbar(im, ax=ax)
    return im


def centerCut(field, axis):
    """return a slice of the data at the center for the specified axis"""
    dims = np.shape(field)
    return np.take(field, indices=int(dims[axis] / 2), axis=axis)


def isHarmonic(field, sphericalMask, shellMask):
    """Checks if the extrema of the field are in the shell."""
    fullField = np.multiply(field, sphericalMask)  # [T]
    reducedField = np.multiply(field, shellMask)
    if int(ptpPPM(fullField)) > int(ptpPPM(reducedField)):
        print(
            "ptpPPM of field:",
            ptpPPM(fullField),
            "ptpPPM on surface",
            ptpPPM(reducedField),
        )
        print("Masked field is NOT a harmonic function...")
        return False
    else:
        print(
            "ptpPPM of field:",
            ptpPPM(fullField),
            "ptpPPM on surface",
            ptpPPM(reducedField),
        )
        print("Masked field is harmonic.")
        sizeSpherical = int(np.nansum(sphericalMask))
        sizeShell = int(np.nansum(shellMask))
        print(
            "Reduced size of field from {} to {} ({}%)".format(
                sizeSpherical, sizeShell, int(100 * sizeShell / sizeSpherical)
            )
        )
        return True


def genQmesh(field, resolution):
    """Generate a mesh of quadratic coordinates"""
    mask = np.zeros(np.shape(field))
    xAxis = np.linspace(
        -(np.size(field, 0) - 1) * resolution / 2,
        (np.size(field, 0) - 1) * resolution / 2,
        np.size(field, 0),
    )
    yAxis = np.linspace(
        -(np.size(field, 1) - 1) * resolution / 2,
        (np.size(field, 1) - 1) * resolution / 2,
        np.size(field, 1),
    )
    zAxis = np.linspace(
        -(np.size(field, 2) - 1) * resolution / 2,
        (np.size(field, 2) - 1) * resolution / 2,
        np.size(field, 2),
    )
    xAxis, yAxis, zAxis = np.meshgrid(xAxis, yAxis, zAxis)

    xAxisSquare = np.square(xAxis)
    yAxisSquare = np.square(yAxis)
    zAxisSquare = np.square(zAxis)

    return mask, xAxisSquare, yAxisSquare, zAxisSquare


def genMask(
    field, resolution, diameter=False, shellThickness=False, axis=False, debug=False
):
    """Generate a mask for a spherical shell"""
    mask, xAxisSquare, yAxisSquare, zAxisSquare = genQmesh(field, resolution)
    if (shellThickness != False) and (diameter != False):
        if debug == True:
            print(
                "Creating shell mask. (resolution = {}, diameter = {}, shellThickness = {})".format(
                    resolution, diameter, shellThickness
                )
            )
            print("The shell is added inside the sphere surface!")
        rAxisSquare = xAxisSquare + yAxisSquare + zAxisSquare
        innerRadiusSquare = (diameter / 2 - shellThickness) ** 2
        outerRadiusSquare = (diameter / 2) ** 2
        mask[
            (rAxisSquare <= outerRadiusSquare) & (rAxisSquare >= innerRadiusSquare)
        ] = 1
    mask[mask == 0] = "NaN"
    return mask


def genSphericalMask(field, diameter, resolution):
    """generate spherical mask 
    with >>diameter<<
    for a >>field<< and a given >>resolution<<
    """
    mask, xAxisSquare, yAxisSquare, zAxisSquare = genQmesh(field, resolution)
    mask[xAxisSquare + yAxisSquare + zAxisSquare <= (diameter / 2) ** 2] = 1
    mask[mask == 0] = "NaN"
    return mask


def genSliceMask(field, diameter, resolution, axis="x"):
    """generate mask for a circular slice
    with >>diameter<<
    for a >>field<< and a given >>resolution<<
    Every input variable has to have the same unit (mm or m or ...)
    """
    mask, xAxisSquare, yAxisSquare, zAxisSquare = genQmesh(field, resolution)
    if axis == "z":
        mask[
            (xAxisSquare + yAxisSquare <= (diameter / 2) ** 2) & (zAxisSquare == 0)
        ] = 1
    if axis == "y":
        mask[
            (xAxisSquare + zAxisSquare <= (diameter / 2) ** 2) & (yAxisSquare == 0)
        ] = 1
    if axis == "x":
        mask[
            (yAxisSquare + zAxisSquare <= (diameter / 2) ** 2) & (xAxisSquare == 0)
        ] = 1
    mask[mask == 0] = "NaN"
    return mask


def genEllipseSliceMask(field, a, b, resolution, axis="x"):
    """generate mask for a circulat slice
    with >>diameter<<
    for a >>field<< and a given >>resolution<<
    Every input variable has to have the same unit (mm or m or ...)
    """
    # generate spherical mask
    mask, xAxisSquare, yAxisSquare, zAxisSquare = genQmesh(field, resolution)
    if axis == "z":
        mask[
            (xAxisSquare / (a / 2) ** 2 + yAxisSquare / (b / 2) ** 2 <= 1)
            & (zAxisSquare == 0)
        ] = 1
    elif axis == "y":
        mask[
            (xAxisSquare / (a / 2) ** 2 + zAxisSquare / (b / 2) ** 2 <= 1)
            & (yAxisSquare == 0)
        ] = 1
    elif axis == "x":
        mask[
            (yAxisSquare / (a / 2) ** 2 + zAxisSquare / (b / 2) ** 2 <= 1)
            & (xAxisSquare == 0)
        ] = 1
    mask[mask == 0] = "NaN"
    return mask


def ptpPPM(field):
    """Calculate the peak-to-peak homogeneity in ppm."""
    return 1e6 * (np.nanmax(field) - np.nanmin(field)) / np.nanmean(field)


def saveParameters(parameters, folder):
    """Saving a dict to the file parameters.npy .
    If the file exist it is beeing updated, if the parameters are not stored already.
    
    __future__: Fix usecase: Some parameters are in dict which are identical to the 
                stored ones and some are new!
    """
    try:
        print("Saving parameters to file...", end=" ")
        print("\x1b[6;30;42m", *parameters.keys(), "\x1b[0m", end=" ")
        oldParameters = loadParameters(folder)
        if parameters.items() <= oldParameters.items():
            print("  ... the parameters are already saved and identical.")
        elif set(parameters).issubset(
            set(oldParameters)
        ):  # here just keys are compared!
            print(
                "  ...\x1b[6;37;41m"
                + " parameters are NOT saved. Other parameters are stored. Please cleanup! "
                + "\x1b[0m"
            )
        else:
            oldParameters.update(parameters)
            np.save(folder + "/parameters", oldParameters)
            print("  ... added.")
    except FileNotFoundError or AttributeError:
        np.save(folder + "/parameters", parameters)
        oldParameters = parameters
    # print('The following parameters are currently stored:\n', *oldParameters.keys())


def loadParameters(folder):
    return np.load(folder + "/parameters.npy", allow_pickle=True).item()


def loadParameter(key, folder):
    return loadParameters(folder)[key]


def displayParameters(folder):
    print(loadParameters(folder))


def createShimfieldsShimRingV2(
    numMagnets=(32, 44),
    rings=4,
    radii=(0.074, 0.097),
    zRange=(-0.08, -0.039, 0.039, 0.08),
    resolution=1000,
    kValue=2,
    simDimensions=(0.04, 0.04, 0.04),
    numRotations=2,
):
    """ Calculating the magnetic field distributions for a single or multiple Halbach Rings.
        This has to be multiplied with the magnetic moment amplitude of a magnet to get the real distribution
        For every magnet position we set 4 different rotations: 0°, 45°, 90°, 135°. This has to be considered in the cost function 
        otherwise two magnets are placed in one position

        resolution is the amount of sample points times data points in one dimension
    """
    mu = mu_0

    # positioning of the magnets in a circle
    if len(zRange) == 2:
        rings = np.linspace(zRange[0], zRange[1], rings)
    elif rings == len(zRange):
        rings = np.array(zRange)
    else:
        print("No clear definition how to place shims...")
    rotation_elements = np.linspace(0, np.pi, numRotations, endpoint=False)

    # create array to store field data
    count = 0
    if type(numMagnets) in (list, tuple):
        totalNumMagnets = np.sum(numMagnets) * np.size(rings) * numRotations
    else:
        totalNumMagnets = numMagnets * np.size(rings) * numRotations * len(radii)
    print(totalNumMagnets, numMagnets, np.size(rings), np.size(numRotations))
    shimFields = np.zeros(
        (
            int(simDimensions[0] * resolution) + 1,
            int(simDimensions[1] * resolution) + 1,
            int(simDimensions[2] * resolution) + 1,
            3,
            totalNumMagnets,
        ),
        dtype=np.float32,
    )

    for rotation in rotation_elements:

        # create halbach array
        for row in rings:
            for i, radius in enumerate(radii):
                angle_elements = np.linspace(
                    -np.pi, np.pi, numMagnets[i], endpoint=False
                )
                for angle in angle_elements:
                    print(
                        "Simulating magnet "
                        + str(count + 1)
                        + " of "
                        + str(totalNumMagnets),
                        end="\t",
                    )
                    position = (row, radius * np.cos(angle), radius * np.sin(angle))

                    print(
                        "@ position {:2.2},\t {:2.2},\t {:2.2}".format(*position),
                        end="\r",
                    )
                    angle = kValue * angle + rotation
                    dip_vec = [0, np.sin(angle), -np.cos(angle)]

                    dip_vec = np.multiply(dip_vec, mu)
                    dip_vec = np.divide(dip_vec, 4 * np.pi)

                    # create mesh coordinates
                    x = np.linspace(
                        -simDimensions[0] / 2 + position[0],
                        simDimensions[0] / 2 + position[0],
                        int(simDimensions[0] * resolution) + 1,
                        dtype=np.float32,
                    )
                    y = np.linspace(
                        -simDimensions[1] / 2 + position[1],
                        simDimensions[1] / 2 + position[1],
                        int(simDimensions[1] * resolution) + 1,
                        dtype=np.float32,
                    )
                    z = np.linspace(
                        -simDimensions[2] / 2 + position[2],
                        simDimensions[2] / 2 + position[2],
                        int(simDimensions[2] * resolution) + 1,
                        dtype=np.float32,
                    )

                    x, y, z = np.meshgrid(x, y, z)

                    vec_dot_dip = 3 * (y * dip_vec[1] + z * dip_vec[2])

                    # calculate the distance of each mesh point to magnet, optimised for speed
                    # for improved memory performance move in to b0 calculations
                    vec_mag = np.square(x) + np.square(y) + np.square(z)
                    # if the magnet is in the origin, we divide by 0, therefore we set it to nan to
                    # avoid getting and error. if this has any effect on speed just leave it out
                    # as we do not care about the values outside of the FOV and even less inside the magnets
                    vec_mag[(vec_mag <= 1e-15) & (vec_mag >= -1e-15)] = "NaN"
                    vec_mag_3 = np.power(vec_mag, 1.5)
                    vec_mag_5 = np.power(vec_mag, 2.5)
                    del vec_mag

                    # calculate contributions of magnet to total field, dipole always points in yz plane
                    # so second term is zero for the x component
                    shimFields[:, :, :, 0, count] = np.divide(
                        np.multiply(x, vec_dot_dip), vec_mag_5
                    )
                    shimFields[:, :, :, 1, count] = np.divide(
                        np.multiply(y, vec_dot_dip), vec_mag_5
                    ) - np.divide(dip_vec[1], vec_mag_3)
                    shimFields[:, :, :, 2, count] = np.divide(
                        np.multiply(z, vec_dot_dip), vec_mag_5
                    ) - np.divide(dip_vec[2], vec_mag_3)
                    count += 1
    print(
        "All magnets are simulated, the shim field array has shape:",
        np.shape(shimFields),
        "\t\t\t",
    )
    return shimFields.swapaxes(
        0, 1
    )  # using i,j indexing as the other is too confusing....


def createShimfieldsDoubleRings(
    numMagnets=72,
    rings=1,
    radii=(0.115, 0.12),
    zRange=(0, 0),
    resolution=1000,
    kValue=2,
    simDimensions=(0.04, 0.04, 0.04),
    numRotations=4,
):
    """ Calculating the magnetic field distributions for a single or multiple Halbach Rings.
        This has to be multiplied with the magnetic moment amplitude of a magnet to get the real distribution
        For every magnet position we set 4 different rotations: 0°, 45°, 90°, 135°. This has to be considered in the cost function 
        otherwise two magnets are placed in one position

        resolution is the amount of sample points times data points in one dimension
    """
    mu = mu_0

    # positioning of the magnets in a circle
    if len(zRange) == 2:
        rings = np.linspace(zRange[0], zRange[1], rings)
    elif rings == len(zRange):
        rings = np.array(zRange)
    else:
        print("No clear definition how to place shims...")
    rotation_elements = np.linspace(0, np.pi, numRotations, endpoint=False)

    # create array to store field data
    count = 0
    totalNumMagnets = numMagnets * np.size(rings) * numRotations * len(radii)
    print(totalNumMagnets, numMagnets, np.size(rings), np.size(numRotations))
    shimFields = np.zeros(
        (
            int(simDimensions[0] * resolution) + 1,
            int(simDimensions[1] * resolution) + 1,
            int(simDimensions[2] * resolution) + 1,
            3,
            totalNumMagnets,
        ),
        dtype=np.float32,
    )

    for rotation in rotation_elements:
        angle_elements = np.linspace(-np.pi, np.pi, numMagnets, endpoint=False)

        # create halbach array
        for row in rings:
            for angle in angle_elements:
                for radius in radii:
                    print(
                        "Simulating magnet "
                        + str(count + 1)
                        + " of "
                        + str(totalNumMagnets),
                        end="\t",
                    )

                    position = (row, radius * np.cos(angle), radius * np.sin(angle))

                    print(
                        "@ position {:2.2},\t {:2.2},\t {:2.2}".format(*position),
                        end="\r",
                    )
                    angle = kValue * angle + rotation
                    dip_vec = [0, np.sin(angle), -np.cos(angle)]

                    dip_vec = np.multiply(dip_vec, mu)
                    dip_vec = np.divide(dip_vec, 4 * np.pi)

                    # create mesh coordinates
                    x = np.linspace(
                        -simDimensions[0] / 2 + position[0],
                        simDimensions[0] / 2 + position[0],
                        int(simDimensions[0] * resolution) + 1,
                        dtype=np.float32,
                    )
                    y = np.linspace(
                        -simDimensions[1] / 2 + position[1],
                        simDimensions[1] / 2 + position[1],
                        int(simDimensions[1] * resolution) + 1,
                        dtype=np.float32,
                    )
                    z = np.linspace(
                        -simDimensions[2] / 2 + position[2],
                        simDimensions[2] / 2 + position[2],
                        int(simDimensions[2] * resolution) + 1,
                        dtype=np.float32,
                    )

                    x, y, z = np.meshgrid(x, y, z)

                    vec_dot_dip = 3 * (y * dip_vec[1] + z * dip_vec[2])

                    # calculate the distance of each mesh point to magnet, optimised for speed
                    # for improved memory performance move in to b0 calculations
                    vec_mag = np.square(x) + np.square(y) + np.square(z)
                    # if the magnet is in the origin, we divide by 0, therefore we set it to nan to
                    # avoid getting and error. if this has any effect on speed just leave it out
                    # as we do not care about the values outside of the FOV and even less inside the magnets
                    vec_mag[(vec_mag <= 1e-15) & (vec_mag >= -1e-15)] = "NaN"
                    vec_mag_3 = np.power(vec_mag, 1.5)
                    vec_mag_5 = np.power(vec_mag, 2.5)
                    del vec_mag

                    # calculate contributions of magnet to total field, dipole always points in yz plane
                    # so second term is zero for the x component
                    shimFields[:, :, :, 0, count] = np.divide(
                        np.multiply(x, vec_dot_dip), vec_mag_5
                    )
                    shimFields[:, :, :, 1, count] = np.divide(
                        np.multiply(y, vec_dot_dip), vec_mag_5
                    ) - np.divide(dip_vec[1], vec_mag_3)
                    shimFields[:, :, :, 2, count] = np.divide(
                        np.multiply(z, vec_dot_dip), vec_mag_5
                    ) - np.divide(dip_vec[2], vec_mag_3)
                    count += 1
    print(
        "All magnets are simulated, the shim field array has shape:",
        np.shape(shimFields),
        "\t\t\t",
    )
    return shimFields.swapaxes(
        0, 1
    )  # using i,j indexing as the other is too confusing....


def createShimfields(
    numMagnets=72,
    rings=1,
    radius=0.115,
    zRange=(0, 0),
    resolution=1000,
    kValue=2,
    simDimensions=(0.04, 0.04, 0.04),
    numRotations=4,
):
    """ Calculating the magnetic field distributions for a single or multiple Halbach Rings.
        This has to be multiplied with the magnetic moment amplitude of a magnet to get the real distribution
        For every magnet position we set 4 different rotations: 0°, 45°, 90°, 135°. This has to be considered in the cost function 
        otherwise two magnets are placed in one position

        resolution is the amount of sample points times data points in one dimension
    """
    mu_0 = mu

    # positioning of the magnets in a circle
    if len(zRange) == 2:
        rings = np.linspace(zRange[0], zRange[1], rings)
    elif rings == len(zRange):
        rings = np.array(zRange)
    else:
        print("No clear definition how to place shims...")
    rotation_elements = np.linspace(0, np.pi, numRotations, endpoint=False)

    # create array to store field data
    count = 0
    totalNumMagnets = numMagnets * np.size(rings) * numRotations
    print(totalNumMagnets, numMagnets, np.size(rings), np.size(numRotations))
    shimFields = np.zeros(
        (
            int(simDimensions[0] * resolution) + 1,
            int(simDimensions[1] * resolution) + 1,
            int(simDimensions[2] * resolution) + 1,
            3,
            totalNumMagnets,
        ),
        dtype=np.float32,
    )

    for rotation in rotation_elements:
        angle_elements = np.linspace(-np.pi, np.pi, numMagnets, endpoint=False)

        # create halbach array
        for row in rings:
            for angle in angle_elements:
                print(
                    "Simulating magnet "
                    + str(count + 1)
                    + " of "
                    + str(totalNumMagnets),
                    end="\t",
                )
                position = (row, radius * np.cos(angle), radius * np.sin(angle))

                print(
                    "@ position {:2.2},\t {:2.2},\t {:2.2}".format(*position), end="\r"
                )
                angle = kValue * angle + rotation
                dip_vec = [0, np.sin(angle), -np.cos(angle)]

                dip_vec = np.multiply(dip_vec, mu)
                dip_vec = np.divide(dip_vec, 4 * np.pi)

                # create mesh coordinates
                x = np.linspace(
                    -simDimensions[0] / 2 + position[0],
                    simDimensions[0] / 2 + position[0],
                    int(simDimensions[0] * resolution) + 1,
                    dtype=np.float32,
                )
                y = np.linspace(
                    -simDimensions[1] / 2 + position[1],
                    simDimensions[1] / 2 + position[1],
                    int(simDimensions[1] * resolution) + 1,
                    dtype=np.float32,
                )
                z = np.linspace(
                    -simDimensions[2] / 2 + position[2],
                    simDimensions[2] / 2 + position[2],
                    int(simDimensions[2] * resolution) + 1,
                    dtype=np.float32,
                )

                x, y, z = np.meshgrid(x, y, z)

                vec_dot_dip = 3 * (y * dip_vec[1] + z * dip_vec[2])

                # calculate the distance of each mesh point to magnet, optimised for speed
                # for improved memory performance move in to b0 calculations
                vec_mag = np.square(x) + np.square(y) + np.square(z)
                # if the magnet is in the origin, we divide by 0, therefore we set it to nan to
                # avoid getting and error. if this has any effect on speed just leave it out
                # as we do not care about the values outside of the FOV and even less inside the magnets
                vec_mag[(vec_mag <= 1e-15) & (vec_mag >= -1e-15)] = "NaN"
                vec_mag_3 = np.power(vec_mag, 1.5)
                vec_mag_5 = np.power(vec_mag, 2.5)
                del vec_mag

                # calculate contributions of magnet to total field, dipole always points in yz plane
                # so second term is zero for the x component
                shimFields[:, :, :, 0, count] = np.divide(
                    np.multiply(x, vec_dot_dip), vec_mag_5
                )
                shimFields[:, :, :, 1, count] = np.divide(
                    np.multiply(y, vec_dot_dip), vec_mag_5
                ) - np.divide(dip_vec[1], vec_mag_3)
                shimFields[:, :, :, 2, count] = np.divide(
                    np.multiply(z, vec_dot_dip), vec_mag_5
                ) - np.divide(dip_vec[2], vec_mag_3)
                count += 1
    print(
        "All magnets are simulated, the shim field array has shape:",
        np.shape(shimFields),
        "\t\t\t",
    )
    return shimFields.swapaxes(
        0, 1
    )  # using i,j indexing as the other is too confusing....

# @njit # this can increase calculation time significantly
# # the individual in the genetic algorithm needs to be changed to
# # a numpy array, check "One Max Problem: Using Numpy" in the deap documentation!
def dna2vector(dna, dipolMoments, numRotations, numMagnets):
    """Casts structured *dna to shim *vector*.
    
    The *dna* is structured each element of the dna vector
    stands for one magnet position and its value represents the type.
    The values of each element are structured in the following way:
        0 -------------> No magnet, 
        1 -------------> Magnet of type 1 rotated by 0°
        ...
        2*rotations ---> Magnet of type 1 rotated to the last angle before 360°
        2*rotations+1 -> Magnet of type 2 rotated by 0°
        ...                     *(Type x stands for Magnet with dipol moment x)
    
    The resulting *vector* will contain the dipol strengths as values and the rotations
    will be written successively, meaning first all magnets / dipol strengths with 0° 
    deviation from the Halbach Dipole orientation will be written, then the first rotation
    and so on...
    
    *dipolMoments* is a list of all dipole moments ordered in the same order as the dna.
    This list should only contain positive values!
    
    *numRotations* is the number of rotations of the magnets possible in one half circle,
    so there are in total 2*rotations possible for each magnet.
    
    *numMagnets* is the number of possible magnets to be placed.
    """
    vector = [0] * numMagnets * numRotations  # np.zeros((len(dna))*numRotations)
    for magnetPos, gene in enumerate(dna):
        if gene != 0:
            magnetType, rotation = divmod((gene - 1), (2 * numRotations))
            if rotation >= numRotations:
                sign = -1
                rotation = rotation % numRotations
            else:
                sign = 1
            index = int(magnetPos + rotation * numMagnets)
            vector[index] = sign * dipolMoments[magnetType]
    return vector


def saveResults(parameters, shimmedField, folder):
    counter = 0
    filename = folder + "/results{}.npy"
    while isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
    np.save(filename, parameters)
    np.save(folder + "/shimmedField{}".format(counter), shimmedField)


def initSimulation(name):
    if not exists(name):
        makedirs(name)
        print("New folder", name, "created.")


def importComsol(filename):
    """Imports 3D Grid Data from Comsol Multiphysics
    Export the data with Comsol in the following manner:
    Expressions: Bx, By, Bz, Bmean
    Output:      File type: Text 
    Data format: Spreadsheet
    For "Points to evaluate in" you have two options:
    a) Grid: use range(start, step, end) with the same value
       of step for each direction
    b) Regular Grid: The number of points for each dimension 
       should be such that the resolution is equal in each direction
    """

    raw = np.loadtxt(filename, skiprows=9, delimiter=",")

    x = raw[:, 0]
    y = raw[:, 1]
    z = raw[:, 2]
    Bx = raw[:, 3]
    By = raw[:, 4]
    Bz = raw[:, 5]
    Bnorm = raw[:, 6]

    def getRes(x):
        res = np.abs(np.unique(x)[1] - np.unique(x)[0])
        return res

    def getShift(x):
        shift = x[np.argmin(np.abs(x))]
        return shift

    res = (getRes(x), getRes(y), getRes(z))
    shift = (getShift(x), getShift(y), getShift(z))

    xInd = np.array((x - shift[0]) / res[0], dtype=int)
    yInd = np.array((y - shift[1]) / res[1], dtype=int)
    zInd = np.array((z - shift[2]) / res[2], dtype=int)

    xInd -= np.min(xInd)
    yInd -= np.min(yInd)
    zInd -= np.min(zInd)

    dims = (np.unique(x).shape[0], np.unique(y).shape[0], np.unique(z).shape[0])
    data = np.zeros((dims))
    data[data == 0] = "NaN"

    for i in range(len(xInd)):
        data[xInd[i], yInd[i], zInd[i]] = Bz[i]

    # change ij indexing to xy indexing -> see numpy meshgrid documentation
    data = data  # .swapaxes(0,1)
    try:
        info = np.loadtxt(filename, skiprows=7, max_rows=1, dtype=np.str)[1:]
        try:
            print(
                *info,
                "\nResolution x: {0} {3}, y: {1} {3}, z: {2} {3}".format(*res, info[2]),
            )
        except IndexError:
            print(info)
    except TypeError:
        print("Update your numpy to have nice output.")

    return data, np.mean(res)


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
