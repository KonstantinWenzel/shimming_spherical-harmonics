import sys
import argparse
import random
import time
import pyshtools

import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator
from deap import creator, base, tools
from os import makedirs
from os.path import exists
from scipy.constants import mu_0

from shimmingFunctions import (
    ptpPPM,
    dna2vector,
    initSimulation,
    loadParameter,
    saveParameters,
    saveResults,
    loadParameters,
)


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
    print("Check if the imported Comsol array is rotated right or flipped.")
    data = data  # .swapaxes(0,1)
    try:
        info = np.loadtxt(filename, skiprows=7, max_rows=1, dtype=np.str)[1:]
        try:
            print(
                *info,
                "\nResolution x: {0} {3}, y: {1} {3}, z: {2} {3}".format(*res, info[2])
            )
        except IndexError:
            print(info)
    except TypeError:
        print("Update your numpy to have nice output.")

    return data, np.mean(res)


def readData(filename, phiNumber, thetaNumber, encoding="lakeshore", axis=None):
    """Depending on the encoding of your data you may want to change this function."""
    print("Reading", filename)

    # Read .dam file (magnetic field camera specific format)
    if encoding == "dam":
        with open(filename, "r") as ins:
            fielddata = []
            for line in ins:
                if line.strip().endswith(" 70") or line.strip().endswith(
                    " 73"
                ):  # Select the lines containing data
                    fielddata.append((line.strip().split()[1].replace(",", ".")))
                elif line.strip().endswith("69"):
                    print("Caution: Only 69 valid cycles.")
                    fielddata.append(line.strip().split()[1])

    # Read lakeshore encoding
    if encoding == "lakeshore":
        with open(filename, "r") as ins:
            fielddata = []
            for line in ins:
                if axis == "x":
                    B = line.split(",")[0]
                elif axis == "y":
                    B = line.split(",")[1]
                elif axis == "z":
                    B = line.split(",")[2]
                else:
                    B = line.split(",")[1]
                    # B = line.split(',')[3]
                    print("Please declare axis!!! y axis chosen for backwards comp.")
                fielddata.append(B)

    fielddata = np.array(fielddata, dtype=float)

    # for debugging:
    # to test the coordinate system use the following to find the upper triangle.
    # fielddata = np.arange(len(fielddata))
    # for i in range(10):
    # for j in range(5):
    # fielddata[40*i + j] = 1000

    fielddata = np.reshape(
        fielddata, (thetaNumber, phiNumber)
    )  # could be and probably is wrong order for lakeshore! fixed 6 lines below

    if encoding == "dam":
        fielddata = fielddata * 1e6  # Convert to Hz
    if encoding == "lakeshore":
        fielddata = fielddata.T  # Transpose data (maybe flip? see 6 lines above)
        fielddata = 1e-3 * fielddata
        fielddata[:, :] = fielddata[::-1, :]
        fielddata = np.roll(fielddata, shift=int(phiNumber / 4), axis=0)
        # fielddata[:10, :10] = 0              # find upper triangle
    f_mean = np.mean(np.ravel(fielddata))  # Average frequency

    return fielddata, f_mean


def calcAngles(phiNumber, thetaNumber):
    """returns the angles of phi and theta for a given number of phi and theta angles
    the theta angles are set to be gauss legendre points and theta are equidistant"""
    samplePoints, weights = np.polynomial.legendre.leggauss(thetaNumber)
    samplePoints = samplePoints[::-1]
    weights = weights[::-1]

    thetaRad = np.arccos(samplePoints) - np.pi / 2

    phiMin = 0  # [rad] Longitude limits
    phiMax = 2 * np.pi * (1 - 1 / phiNumber)
    phiRad = np.linspace(phiMin, phiMax, phiNumber)

    return phiRad, thetaRad


def toCathesianCoords(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def interpolMeshToSphere(field, resolution, phiRad, thetaRad, radius, _min, _max):
    """interpolatates a field with a given resolution on a sphere"""
    xAxis = np.linspace(_min, _max, field.shape[0])
    yAxis = np.linspace(_min, _max, field.shape[0])
    zAxis = np.linspace(_min, _max, field.shape[0])
    interpol = RegularGridInterpolator((xAxis, yAxis, zAxis), field)
    fielddata = []
    for _phi in phiRad:
        for _theta in thetaRad:
            # theta must be in rotated as co or- not co- (latitude longitude)
            fielddata.append(
                interpol(toCathesianCoords(radius, _phi, _theta + np.pi / 2))
            )

    fielddata = np.array(fielddata, dtype=float)
    fielddata = np.reshape(fielddata, (len(phiRad), len(thetaRad)))
    return fielddata


def integrate(integrand, phiRad, thetaNumber):
    """Integrates a descrete function
    phiRad          - the phi angles the integrand is sampled on
    thetaNumber     - the number of gauss legendre points in theta
    the theta angles are set by gauss legendre

    the integral along axis 1 is with gauss legendre
    and along axis 2 with trapzoids
    """

    # calculate gauss-legendre points
    samplePoints, weights = leggauss(thetaNumber)
    # padd integrand
    integrand_padded = np.vstack((integrand, integrand[0]))
    # calculate phis for trapz
    phis = np.append(phiRad, phiRad[-1] + phiRad[1] - phiRad[0])
    # integrate over phi with trapezoidal rule
    integration1 = np.trapz(integrand_padded, x=phis, axis=0)
    # integrate over theta with gauss legendre
    integration2 = np.dot(integration1, weights)
    return integration2


def calcMaxSHdegree(thetaNumber, phiNumber):
    """calculating the maximum spherical harmonic coefficients
    based on the number of sampling points"""
    lmax = int((thetaNumber - 1) / 2 - 1)
    mmax = int(phiNumber / 2 - 1)
    return lmax, mmax


def calcSHcoefficents(field, lat, lon, phiRad, thetaNumber, lmax, mmax, threshold=None):
    """returns the sh coefficients of a given field sampled at lat, lon
    as a pyshtools class"""
    # initialize pyshtools
    clmRecon = pyshtools.SHCoeffs.from_zeros(lmax=lmax, normalization="ortho")

    for l in range(lmax):
        for m in range(-l, l + 1):
            if abs(m) <= mmax:
                clm = pyshtools.SHCoeffs.from_zeros(lmax=lmax, normalization="ortho")
                clm.set_coeffs(1.0, l, m)
                Ylm = clm.expand(lat=lat, lon=lon)
                coeff = integrate(Ylm * field, phiRad, thetaNumber)
                if threshold != None:
                    if abs(coeff) < threshold:
                        clmRecon.set_coeffs(0, l, m)
                    else:
                        clmRecon.set_coeffs(coeff, l, m)
                else:
                    clmRecon.set_coeffs(coeff, l, m)
    return clmRecon


def calcDipolMomentAnalytical(remanence, volume):
    """Calculating the magnetic moment from the remanence in T and the volume in m^3"""
    m = remanence * volume / mu_0  # [A * m^2]
    return m


def calcFieldpointOneMagnetZ(x0, y0, z0, r, phi, theta, mx, my, mz):
    """ok if we do it like this we need one function which returns for a given angle and
    a given radius for a magnet at postion x,y,z the field"""
    mu = mu_0

    x = r * np.sin(theta) * np.cos(phi) - x0
    y = r * np.sin(theta) * np.sin(phi) - y0
    z = r * np.cos(theta) - z0

    dip_vec = (mx, my, mz)

    dip_vec = np.multiply(dip_vec, mu)
    dip_vec = np.divide(dip_vec, 4 * np.pi)

    vec_dot_dip = 3 * (x * dip_vec[0] + y * dip_vec[1] + z * dip_vec[2])

    vec_mag = np.square(x) + np.square(y) + np.square(z)
    vec_mag_3 = np.power(vec_mag, 1.5)
    vec_mag_5 = np.power(vec_mag, 2.5)
    del vec_mag

    # standard notation z || B0, bore along y axis
    # shimFields = np.divide(np.multiply(z, vec_dot_dip), vec_mag_5) - np.divide(dip_vec[2],vec_mag_3)
    # rotated notation y || By, bore along z axis

    shimFields = np.divide(np.multiply(y, vec_dot_dip), vec_mag_5) - np.divide(
        dip_vec[1], vec_mag_3
    )

    return shimFields


def calcFieldOneMagnetZ(x0, y0, z0, r, mx, my, mz, phiRad, thetaRad, remanence, size):
    """calculating the field of one magnet sampled on a sphere"""
    fielddata = []
    for _phi in phiRad:
        for _theta in thetaRad:
            Bz = calcFieldpointOneMagnetZ(
                x0, y0, z0, r, _phi, _theta + np.pi / 2, mx, my, mz
            )
            Bz *= calcDipolMomentAnalytical(remanence, size ** 3)
            fielddata.append(Bz)

    fielddata = np.array(fielddata, dtype=float)
    fielddata = np.reshape(fielddata, (len(phiRad), len(thetaRad)))
    return fielddata


def calcSHarrayLength(phiRad, thetaRad, lat, lon, lmax, mmax):
    """really ugly implementation to calculate the size of this array
    replace me with a simple equation!"""
    field = calcFieldOneMagnetZ(1e10, 1e10, 1e10, 0, 1, 0, 0, phiRad, thetaRad, 1, 1)
    coeffs = calcSHcoefficents(
        field, lat, lon, phiRad, len(thetaRad), lmax, mmax
    ).to_array()
    print(np.shape(coeffs))
    return np.shape(coeffs)


def calcSHcoefficentsShim(params, lmax, mmax, threshold=None):
    """Calculating the magnetic field distributions for a single or multiple Halbach Rings.
    This has to be multiplied with the magnetic moment amplitude of a magnet to get the real distribution
    For every magnet position we set 4 different rotations: 0°, 45°, 90°, 135°. This has to be considered in the cost function
    otherwise two magnets are placed in one position"""
    xRange = params["xRange"]
    numRings = params["numRings"]
    numRotations = params["numRotations"]
    # lmax, mmax = calcMaxSHdegree(params['thetaNumber'], params['phiNumber'])
    thetaNumber = params["thetaNumber"]
    size = params["size"]
    remanence = params["remanence"]
    numMagnets = params["numMagnets"]
    kValue = params["kValue"]
    r = params["FOV diameter"] / 2.0
    radius = params["radius"]
    if "numMagnets2" in params:
        numMagnets = [numMagnets, params["numMagnets2"]]
        radii = [radius, params["radius2"]]
    else:
        radii = [radius]
        numMagnets = [numMagnets]

    coeffsShape = calcSHarrayLength(phiRad, thetaRad, lat, lon, lmax, mmax)
    print("phiRad and thetaRad are not local here....")
    print("find shape of coeffs")

    # positioning of the magnets in a circle
    if len(xRange) == 2:
        rings = np.linspace(xRange[0], xRange[1], numRings)
    elif len(xRange) == numRings:
        rings = xRange
    else:
        print("Number of shim rings do not fit to xRange.")
    rotation_elements = np.linspace(0, np.pi, numRotations, endpoint=False)

    # create array to store field data
    count = 0
    totalNumMagnets = np.sum(numMagnets) * np.size(rings) * numRotations
    shimCoeffs = np.zeros((*coeffsShape, totalNumMagnets))  # , dtype=np.float32)
    print(shimCoeffs.shape, "\n")

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

                    # standard notation z || B0, bore along y axis
                    # position = (row, radius*np.cos(angle), radius*np.sin(angle))
                    # rotated notation y || By, bore along z axis
                    position = (radius * np.cos(angle), radius * np.sin(angle), row)

                    print(
                        "@ position {:2.2},\t {:2.2},\t {:2.2}\t\t\t".format(*position),
                        end="\r",
                    )
                    angle = kValue * angle + rotation

                    # standard notation z || B0, bore along y axis
                    # dip_vec = [0, np.sin(angle), -np.cos(angle)]
                    # rotated notation y || By, bore along z axis
                    dip_vec = [np.sin(angle), -np.cos(angle), 0]

                    x0, y0, z0 = position
                    mx, my, mz = dip_vec

                    field = calcFieldOneMagnetZ(
                        x0, y0, z0, r, mx, my, mz, phiRad, thetaRad, remanence, size
                    )
                    shimCoeffs[..., count] = calcSHcoefficents(
                        field,
                        lat,
                        lon,
                        phiRad,
                        thetaNumber,
                        lmax,
                        mmax,
                        threshold=threshold,
                    ).to_array()

                    count += 1
    print(
        "All magnets are simulated, the shim field array has shape:",
        np.shape(shimCoeffs),
        "\t\t\t",
    )
    return shimCoeffs


def calcFieldsShim(params):
    """Calculating the magnetic field distributions for a single or multiple Halbach Rings.
    This has to be multiplied with the magnetic moment amplitude of a magnet to get the real distribution
    For every magnet position we set 4 different rotations: 0°, 45°, 90°, 135°. This has to be considered in the cost function
    otherwise two magnets are placed in one position"""
    xRange = params["xRange"]
    numRings = params["numRings"]
    numRotations = params["numRotations"]
    # lmax, mmax = calcMaxSHdegree(params['thetaNumber'], params['phiNumber'])
    thetaNumber = params["thetaNumber"]
    phiNumber = params["phiNumber"]
    size = params["size"]
    remanence = params["remanence"]
    numMagnets = params["numMagnets"]
    kValue = params["kValue"]
    r = params["FOV diameter"] / 2.0
    radius = params["radius"]
    if "numMagnets2" in params:
        numMagnets = [numMagnets, params["numMagnets2"]]
        radii = [radius, params["radius2"]]
    else:
        radii = [radius]
        numMagnets = [numMagnets]

    coeffsShape = calcSHarrayLength(phiRad, thetaRad, lat, lon, lmax, mmax)
    print("phiRad and thetaRad are not local here....")
    print("find shape of coeffs")

    # positioning of the magnets in a circle
    if len(xRange) == 2:
        rings = np.linspace(xRange[0], xRange[1], numRings)
    elif len(xRange) == numRings:
        rings = xRange
    else:
        print("Number of shim rings do not fit to xRange.")
    rotation_elements = np.linspace(0, np.pi, numRotations, endpoint=False)

    # create array to store field data
    count = 0
    totalNumMagnets = np.sum(numMagnets) * np.size(rings) * numRotations
    # totalNumMagnets = numMagnets*np.size(rings)*numRotations
    shimFields = np.zeros(
        (phiNumber, thetaNumber, totalNumMagnets)
    )  # , dtype=np.float32)
    print("size of shim fields")
    print(shimFields.shape)

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

                    # standard notation z || B0, bore along y axis
                    # position = (row, radius*np.cos(angle), radius*np.sin(angle))
                    # rotated notation y || By, bore along z axis
                    position = (radius * np.cos(angle), radius * np.sin(angle), row)

                    print(
                        "@ position {:2.2},\t {:2.2},\t {:2.2}\t\t\t".format(*position),
                        end="\r",
                    )
                    angle = kValue * angle + rotation

                    # standard notation z || B0, bore along y axis
                    # dip_vec = [0, np.sin(angle), -np.cos(angle)]
                    # rotated notation y || By, bore along z axis
                    dip_vec = [np.sin(angle), -np.cos(angle), 0]

                    x0, y0, z0 = position
                    mx, my, mz = dip_vec

                    field = calcFieldOneMagnetZ(
                        x0, y0, z0, r, mx, my, mz, phiRad, thetaRad, remanence, size
                    )
                    shimFields[..., count] = field

                    count += 1
    return shimFields


def flatten_and_crop_coeff_array(coeffs, lmax, mmax):
    """flattens and cropping zeros of array to increase speed of the algorithm"""
    cropped_coeffs = []
    shape = coeffs.shape
    for i in range(shape[0]):
        for l in range(1, shape[1]):  # leaving out first SH coeff
            for m in range(shape[2]):
                # if (j >= k) & (j < shape[1] - 1):
                # if (j >= k):
                if (l >= m) & (l < lmax) & (m <= mmax) & (i - 1 != m):
                    cropped_coeffs += [coeffs[i, l, m]]
                else:
                    if coeffs[i, l, m] != 0:
                        print("Coeffs are not cropped correctly!!!")
    return np.array(cropped_coeffs)


def optimize(b0, A, params, algorithm="GA"):
    """returns the optimal solution vector x of the problem
                    b0 + Ax = const

    GA or LA to find optimum of harmonics
    rather put array or! pyshtools.Coeffs.to_array() do not put pythingi"""
    if algorithm == "GA":
        b0_ = flatten_and_crop_coeff_array(b0, params["lmax"], params["mmax"])
        print(b0_)

        A_ = np.zeros((*b0_.shape, A.shape[-1]))
        for i in range(A_.shape[-1]):
            A_[..., i] = flatten_and_crop_coeff_array(
                A[..., i], params["lmax"], params["mmax"]
            )

        def fieldError(dipolMoments, numRotations, numMagnets, dna):
            """Casts dna to peak to peak deviation of magnetic field"""
            shimVector = dna2vector(dna, dipolMoments, numRotations, numMagnets)
            shimVector = np.array(shimVector)
            tempShimmedField = b0_ + np.matmul(A_, shimVector)
            return (np.sum(np.abs(tempShimmedField)),)

        # Simple Configuration
        popSim = params["popSim"]
        minGeneration = params["minGeneration"]
        maxGeneration = params["maxGeneration"]

        # genetic propabilities
        CXPB = params["CXPB"]  # crossover and mutation
        MUTPB = params["MUTPB"]  # crossover and mutation
        MUTINDPB = params["MUTINDPB"]  # propability for mutation of a gene

        # magnetic properties
        remanence = params["remanence"]
        size = params["size"]

        # other info
        dipolMomentsAreInShimMatrix = params["dipolMomentsAreInShimMatrix"]
        numRotations = params["numRotations"]
        numTotalMagnets = params["numTotalMagnets"]
        numMagnets = params["numMagnets"]
        numShimRings = params["numRings"]

        # number of cost function calls
        costFunctionCalls = np.copy(popSim)

        if dipolMomentsAreInShimMatrix == True:
            numSizes = 1
        numGenes = 2 * numRotations * numSizes  #    # maximal number of value of gene!

        random.seed()

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()

        # define values for individuals
        toolbox.register("genes", random.randint, 0, numGenes)

        # chromosome is defined here, each magnet position is one  gene, so the gene is numTotalMagnets long
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.genes,
            numTotalMagnets,
        )

        # define the population size, the number of different shimVectors in each generation
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        dipolMoments = []
        if dipolMomentsAreInShimMatrix == True:
            dipolMoments += [
                1
            ]  # <- here in field data ##[calcDipolMomentAnalytical(remanence = remanence, volume = (size**3))]
        else:
            for remanence, size in zip(remanences, sizes):
                dipolMoments += [
                    calcDipolMomentAnalytical(remanence=remanence, volume=(size ** 3))
                ]

        toolbox.register(
            "evaluate", fieldError, dipolMoments, numRotations, numTotalMagnets
        )

        # define the evolutionary behaviour
        toolbox.register("mate", tools.cxTwoPoint)

        toolbox.register(
            "mutate", tools.mutUniformInt, low=0, up=numGenes, indpb=MUTINDPB
        )

        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=popSim)

        print("Start of evolution")
        startTime = time.time()
        fitnesses = list(map(toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop), end=" ")
        print("  Evaluation took: " + str(time.time() - startTime) + " seconds")
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        minTracker = []

        startEvolution = time.time()

        bestError = np.inf
        realMin = []

        # Begin the evolution
        while ((g < minGeneration) or (np.argmin(minTracker) > 0.8 * g)) and (
            g < maxGeneration
        ):
            startTime = time.time()
            # A new generation
            g = g + 1
            # print("-- Generation %i --" % g, end=" ")

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # numInvalid = 0
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                costFunctionCalls += 1

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            minTracker += [min(fits)]

            # print("  Min %s" % min(fits), end="")
            estimatedTime = (
                (maxGeneration - g) * (time.time() - startEvolution) / (g + 1)
            )
            estimatedTime = time.strftime("%Hh %Mm %Ss", time.gmtime(estimatedTime))
            estimatedTime2 = (maxGeneration - g) * (time.time() - startTime)
            estimatedTime2 = time.strftime("%Hh %Mm %Ss", time.gmtime(estimatedTime2))

            if min(fits) < bestError:
                # print("\nNEW BEST!")
                bestError = min(fits)
                realMin = tools.selBest(pop, 1)[0]

            if min(fits) != np.inf:
                print(
                    "  Gen {:5} took {:.5} seconds with fitness {:.5}".format(
                        g, str(time.time() - startTime), min(fits)
                    ),
                    end="",
                )
                print(
                    "\t {:.2f}% conv. {:.2f}% (maximum approx. {} to {} left)".format(
                        100 * g / maxGeneration,
                        (125 * np.argmin(minTracker) / g) % 100,
                        estimatedTime,
                        estimatedTime2,
                    ),
                    end="\r",
                )
            else:
                print("Infinity", g * ".", end="\r")

        print("\n-- End of (successful) evolution --")
        totalTime = time.strftime(
            "%Hh %Mm %Ss", time.gmtime(time.time() - startEvolution)
        )
        print(
            "Evolution took "
            + totalTime
            + " and {} cost function calls.".format(costFunctionCalls)
        )

        bestVector = np.array(
            dna2vector(realMin, dipolMoments, numRotations, numTotalMagnets)
        )

        shim = np.matmul(A, bestVector)

        shimmedFieldCoeffs = b0 + shim

        results = {
            "costFunctionCalls": costFunctionCalls,
            "totalTime": totalTime,
            "bestVector": bestVector,
            "realMin": np.array(realMin),
            "Generations": g,
        }

        return shimmedFieldCoeffs, results

    elif algorithm == "GEKKO":
        """Does not work yet."""
        startTime = time.time()
        numRotations = params["numRotations"]

        # flatten A and b0
        # A_ = A[:,1:,1:].reshape(np.prod(np.shape(A[:,1:,1:])[:-1]), np.prod(np.shape(A[:,1:,1:])[-1]))
        # b0_ = b0[:,1:,1:].reshape(np.prod(np.shape(b0[:,1:,1:])))
        b0_ = flatten_and_crop_coeff_array(b0, params["lmax"], params["mmax"])
        A_ = np.zeros((*b0_.shape, A.shape[-1]))
        for i in range(A_.shape[-1]):
            A_[..., i] = flatten_and_crop_coeff_array(
                A[..., i], params["lmax"], params["mmax"]
            )

        m = gekko.GEKKO(remote=False)  # Initialize gekko
        m.options.SOLVER = 1  # APOPT is an MINLP solver, it can handle integers

        print("Solver will not converge... . Solver options are not configured... .")
        # m.solver_options = ['minlp_maximum_iterations 50000', \
        # 'minlp_max_iter_with_int_sol 100', \
        # 'nlp_maximum_iterations 500']
        # optional solver settings with APOPT
        # m.solver_options = ['minlp_maximum_iterations 500', \
        # # minlp iterations with integer solution
        # 'minlp_max_iter_with_int_sol 10', \
        # # treat minlp as nlp
        # 'minlp_as_nlp 0', \
        # # nlp sub-problem max iterations
        # 'nlp_maximum_iterations 50', \
        # # 1 = depth first, 2 = breadth first
        # 'minlp_branch_method 1', \
        # # maximum deviation from whole number
        # 'minlp_integer_tol 0.05', \
        # # covergence tolerance
        # 'minlp_gap_tol 0.01']#

        rescaleFactor = 3  # 3#2#1e1# 1e-3 * 1e4 * 1e-1#e4     # rescale cost function

        k = A_.shape[0]
        l = A_.shape[1]

        x = m.Array(m.Var, l, value=0, lb=-1, ub=1, integer=True)
        Ax = m.Array(m.Var, k)
        obj = m.Var()

        m.Equations(
            [Ax[j] == m.sum([A_[j, i] * x[i] for i in range(l)]) for j in range(k)]
        )
        m.Equation(
            obj == rescaleFactor * m.sum([(Ax[j] + b0_[j]) ** 2 for j in range(k)])
        )
        # m.Equation(obj == rescaleFactor*m.sum([m.abs(Ax[j] + b0_[j]) for j in range(k)]))
        print("initial objective", rescaleFactor * np.sum(b0_ ** 2))
        # print('initial objective', rescaleFactor*np.sum(np.abs(b0_)))

        # restrict state vector
        for i in range(0, l, numRotations):
            m.Equation(1 >= m.sum([x[i + j] ** 2 for j in range(numRotations)]))

        m.Obj(obj)

        m.solve(disp=True, timing=True)  # Solve
        print("Objective: " + str(m.options.objfcnval))
        print("\n")
        print("Equivalent ppm")
        bestVectorGEKKO = np.array([int(x[i].value[0]) for i in range(len(x))])
        print(bestVectorGEKKO)

        shim = np.matmul(A, bestVectorGEKKO)
        shimmedFieldCoeffs = b0 + shim

        totalTime = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - startTime))
        results = {
            #'costFunctionCalls' : costFunctionCalls,
            "totalTime": totalTime,
            "bestVector": bestVectorGEKKO,
            #'realMin' : np.array(realMin),
            #'Generations' : g,
        }

        return shimmedFieldCoeffs, results

    else:
        print(algorithm, "not jet implemented, you can choose:", "GA", "GEKKO")


def reconstructShimmedField(coeffs, lat, lon):
    """Definetly needed!!! """
    # initialize pyshtools
    clmRecon = pyshtools.SHCoeffs.from_array(coeffs, normalization="ortho")
    Y_recon = clmRecon.expand(lat=lat, lon=lon)
    return Y_recon


if __name__ == "__main__":
    # define name
    name = "results/test"

    # command line options can be added with
    # parser = argparse.ArgumentParser()
    # parser.add_argument("size", type=int)
    # args = parser.parse_args()
    # size = args.size

    # if True parameters of prev. sim are loaded,
    # ! all the following parameters will be overwritten
    redo = False
    initSimulation(name)

    # writing all prints to a logfile
    # sys.stdout = open(name + "/log.txt", "w+")

    # define simulation parameters
    parameters = {
        "xRange": (-77.25e-3, -27.25e-3, 27.25e-3, 77.25e-3),
        "numRings": 4,
        "kValue": 1,
        "numRotations": 2,
        "phiNumber": 20 * 2,
        "thetaNumber": 20 * 2,
        "FOV diameter": 0.04,
        "dimension": (0.05, 0.05, 0.05),
        "size": 9e-3,
        "remanence": 1.3,
        "popSim": 25000,
        "minGeneration": 300,
        "maxGeneration": 2000,
        "CXPB": 0.75,  # crossover
        "MUTPB": 0.2,  # mutation
        "MUTINDPB": 0.05,  # propability for mutation of a gene
        "dipolMomentsAreInShimMatrix": True,
        "radius": 0.074,
        "radius2": 0.097,
        "numMagnets": 3, # 2,
        "numMagnets2": 4, # 4,
    }

    parameters.update(
        {
            "numTotalMagnets": np.sum(
                [parameters["numMagnets"], parameters["numMagnets2"]]
            )
            * parameters["numRings"]
        }
    )

    if redo:
        oldname = name[:-1] + str(int(name[-1]) - 1)
        print("redoing", oldname)
        parameters = loadParameters(oldname)

    # save simulation parameters
    saveParameters(parameters, folder=name)

    # define angles
    phiRad, thetaRad = calcAngles(parameters["phiNumber"], parameters["thetaNumber"])

    # currently unused
    # # grid data can be imported and interpolated to the desired points
    # # load Comsol field data
    # fieldCatesian, resolution = importComsol('B0fieldMaps/6ringDesign_IdealField.csv')

    # # interpolate Comsol data on a sphere
    # field = interpolMeshToSphere(
    # fieldCatesian, resolution, phiRad, thetaRad, parameters['FOV diameter']/2,
    # _min= -0.5*parameters['dimension'][0], _max = 0.5*parameters['dimension'][0]
    # )

    # import measured field data
    field, field_mean = readData(
        "2020_06_04_4cmSphere.txt",
        phiNumber=parameters["phiNumber"],
        thetaNumber=parameters["thetaNumber"],
        axis="x",
    )

    # make a mesh of angles for shtools
    phiDeg, thetaDeg = np.rad2deg(phiRad), np.rad2deg(thetaRad)
    lat, lon = np.meshgrid(thetaDeg, phiDeg)  # latitude and longitude for shtools

    # calculate maximum spherical harmonic degrees
    lmax, mmax = calcMaxSHdegree(parameters["thetaNumber"], parameters["phiNumber"])
    lmax = 5  # not including lmax
    mmax = 2  # including mmax
    parameters.update({"lmax": lmax, "mmax": mmax})

    # calculate the SH coefficients of the field as a pyshclass
    fieldCoeffs = calcSHcoefficents(
        field,
        lat,
        lon,
        phiRad,
        parameters["thetaNumber"],
        lmax,
        mmax,
        threshold=None,
    )

    # calculate the SH coefficiens of the shims
    shimFieldCoeffs = calcSHcoefficentsShim(parameters, lmax, mmax)

    # save coeffs
    print("Saving coeffs...")
    if not exists(name + "/coeffs"):
        makedirs(name + "/coeffs")
    fieldCoeffs.to_file(name + "/coeffs" + "/initialFieldCoeffs")
    for i in range(np.shape(shimFieldCoeffs)[-1]):
        pass  # takes forever and does not really add anything
        # saving the coefficients of each individual shim
        _ = pyshtools.SHCoeffs.from_array(
            shimFieldCoeffs[..., i], normalization="ortho"
        )
        _.to_file(name + "/coeffs" + "/shimFieldCoeffs_{}".format(i))
    del i, _
    print("... coeffs saved. They have shape {}".format(np.shape(shimFieldCoeffs)))

    # find the best configuration
    shimmedFieldCoeffs, results = optimize(
        fieldCoeffs.to_array(), shimFieldCoeffs, parameters
    )
    print(results)

    # save results
    pyshtools.SHCoeffs.from_array(shimmedFieldCoeffs, normalization="ortho").to_file(
        name + "/coeffs" + "/coeffsAfterShim"
    )
    ###### end of the main part of the script, we only plot the results below ######

    # plot
    dtheta = thetaRad[1] - thetaRad[0]
    dphi = phiRad[1] - phiRad[0]
    th_m, ph_m = np.meshgrid(
        np.append(
            thetaDeg - np.rad2deg(dtheta) / 2, thetaDeg[-1] + np.rad2deg(dtheta) / 2
        ),
        np.append(phiRad - dphi / 2, phiRad[-1] + dphi / 2),
    )

    fieldReconstructed = fieldCoeffs.expand(lat=lat, lon=lon)

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="polar")  # polar plots
    ax2 = fig.add_subplot(132, projection="polar")
    ax3 = fig.add_subplot(133, projection="polar")
    vmin = 1e3 * np.min([field, fieldReconstructed])
    vmax = 1e3 * np.max([field, fieldReconstructed])
    radarplot1 = ax1.pcolormesh(
        ph_m, th_m, 1e3 * fieldReconstructed, vmin=vmin, vmax=vmax
    )
    radarplot2 = ax2.pcolormesh(ph_m, th_m, 1e3 * field, vmin=vmin, vmax=vmax)
    radarplot3 = ax3.pcolormesh(ph_m, th_m, 1e3 * fieldReconstructed - 1e3 * field)
    cbar1 = plt.colorbar(radarplot1, ax=ax1)
    cbar1.set_label("reconstructed field in mT")
    cbar2 = plt.colorbar(radarplot2, ax=ax2)
    cbar2.set_label("measured field before shim in mT")
    cbar3 = plt.colorbar(radarplot3, ax=ax3)
    cbar3.set_label("difference of field recon \nand measurement in mT")
    plt.tight_layout()
    plt.savefig(name + "/initialField")
    plt.close(fig)

    shimmedField = reconstructShimmedField(shimmedFieldCoeffs, lat, lon)

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="polar")  # polar plots
    ax2 = fig.add_subplot(132, projection="polar")
    ax3 = fig.add_subplot(133, projection="polar")
    radarplot1 = ax1.pcolormesh(ph_m, th_m, 1e3 * (shimmedField))
    radarplot2 = ax2.pcolormesh(ph_m, th_m, 1e3 * (field))
    radarplot3 = ax3.pcolormesh(ph_m, th_m, 1e3 * shimmedField - 1e3 * field)
    cbar1 = plt.colorbar(radarplot1, ax=ax1)
    cbar1.set_label("field after shim in mT")
    cbar2 = plt.colorbar(radarplot2, ax=ax2)
    cbar2.set_label("field before shim in mT")
    cbar3 = plt.colorbar(radarplot3, ax=ax3)
    cbar3.set_label("field of shim in mT")
    plt.tight_layout()
    plt.savefig(name + "/shimmedField")
    plt.close(fig)

    print(
        "max:",
        np.nanmax(shimmedField),
        "T, min:",
        np.nanmin(shimmedField),
        "T, ptp:",
        1e6
        * (np.nanmax(shimmedField) - np.nanmin(shimmedField))
        / np.nanmean(shimmedField),
        "ppm",
    )

    ### with direct calcuation
    shimmedField_noSH = calcFieldsShim(parameters)
    shimmedField_noSH = field + np.matmul(shimmedField_noSH, results["bestVector"])
    np.save(name + "/shimmedField_noSH", shimmedField_noSH)
    np.save(name + "/shimField_noSH", shimmedField_noSH - field)
    np.save(name + "/initialField_noSH", field)
    np.save(name + "/shimedField_withSH", shimmedField)
    np.save(name + "/shimField_withSH", shimmedField - fieldReconstructed)
    np.save(name + "/initialField_withSH", fieldReconstructed)

    results.update(
        {
            "initial homogeneity with SH": 1e6
            * (np.nanmax(fieldReconstructed) - np.nanmin(fieldReconstructed))
            / np.nanmean(fieldReconstructed),
            "shimmed homogeneity with SH": 1e6
            * (np.nanmax(shimmedField) - np.nanmin(shimmedField))
            / np.nanmean(shimmedField),
            "initial homogeneity without SH": 1e6
            * (np.nanmax(field) - np.nanmin(field))
            / np.nanmean(field),
            "shimmed homogeneity without SH": 1e6
            * (np.nanmax(shimmedField_noSH) - np.nanmin(shimmedField_noSH))
            / np.nanmean(shimmedField_noSH),
        }
    )
    np.save(name + "/results.npy", results)

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="polar")  # polar plots
    ax2 = fig.add_subplot(132, projection="polar")
    ax3 = fig.add_subplot(133, projection="polar")
    radarplot1 = ax1.pcolormesh(ph_m, th_m, 1e3 * (shimmedField_noSH - field))
    radarplot2 = ax2.pcolormesh(ph_m, th_m, 1e3 * (shimmedField - fieldReconstructed))
    radarplot3 = ax3.pcolormesh(
        ph_m,
        th_m,
        1e3 * (shimmedField_noSH - field - shimmedField + fieldReconstructed),
    )
    cbar1 = plt.colorbar(radarplot1, ax=ax1)
    cbar1.set_label("shim field in mT \n(recon. from dipol)")
    cbar2 = plt.colorbar(radarplot2, ax=ax2)
    cbar2.set_label("shim field in mT \n(recon. from coeffs)")
    cbar3 = plt.colorbar(radarplot3, ax=ax3)
    cbar3.set_label("deviation of shim fields\nrecon. from coeffs - dipol in mT")
    plt.tight_layout()
    plt.savefig(name + "/shimField_reconVSfull")
    plt.close(fig)

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="polar")  # polar plots
    ax2 = fig.add_subplot(132, projection="polar")
    ax3 = fig.add_subplot(133, projection="polar")
    radarplot1 = ax1.pcolormesh(ph_m, th_m, 1e3 * (shimmedField_noSH))
    radarplot2 = ax2.pcolormesh(ph_m, th_m, 1e3 * (shimmedField_noSH - field))
    radarplot3 = ax3.pcolormesh(ph_m, th_m, 1e3 * field)
    cbar1 = plt.colorbar(radarplot1, ax=ax1)
    cbar1.set_label("field after shim in mT\n(not from coeffs)")
    cbar2 = plt.colorbar(radarplot2, ax=ax2)
    cbar2.set_label("shim field not from coeffs in mT")
    cbar3 = plt.colorbar(radarplot3, ax=ax3)
    cbar3.set_label("initial field in mT\n(not from coeffs)")
    plt.tight_layout()
    plt.savefig(name + "/shimmedField_reconstructedNoSH")
    plt.close(fig)

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="polar")  # polar plots
    ax2 = fig.add_subplot(132, projection="polar")
    ax3 = fig.add_subplot(133, projection="polar")
    radarplot1 = ax1.pcolormesh(ph_m, th_m, 1e3 * (shimmedField))
    radarplot2 = ax2.pcolormesh(ph_m, th_m, 1e3 * (shimmedField - fieldReconstructed))
    radarplot3 = ax3.pcolormesh(ph_m, th_m, 1e3 * fieldReconstructed)
    cbar1 = plt.colorbar(radarplot1, ax=ax1)
    cbar1.set_label("field after shim in mT\n(recon. from coeffs)")
    cbar2 = plt.colorbar(radarplot2, ax=ax2)
    cbar2.set_label("shim field from coeffs in mT")
    cbar3 = plt.colorbar(radarplot3, ax=ax3)
    cbar3.set_label("initial field in mT\n(recon. from coeffs)")
    plt.tight_layout()
    plt.savefig(name + "/shimmedField_reconstructedOnlySH")
    plt.close(fig)

    print(
        "Without SH reconstructed for the given arrangement:\n",
        "max:",
        np.nanmax(shimmedField),
        "T, min:",
        np.nanmin(shimmedField),
        "T, ptp:",
        1e6
        * (np.nanmax(shimmedField) - np.nanmin(shimmedField))
        / np.nanmean(shimmedField),
        "ppm",
    )

    fig, ax = pyshtools.SHCoeffs.from_array(
        fieldCoeffs.to_array(), normalization="ortho"
    ).plot_spectrum(show=False)
    plt.savefig(name + "/initialSpectrum")

    fig, ax = pyshtools.SHCoeffs.from_array(
        fieldCoeffs.to_array(), normalization="ortho"
    ).plot_spectrum2d(show=False)
    plt.savefig(name + "/initialSpectrum2d")

    fig, ax = pyshtools.SHCoeffs.from_array(
        shimmedFieldCoeffs - fieldCoeffs.to_array(), normalization="ortho"
    ).plot_spectrum(show=False)
    plt.savefig(name + "/shimSpectrum")

    fig, ax = pyshtools.SHCoeffs.from_array(
        shimmedFieldCoeffs - fieldCoeffs.to_array(), normalization="ortho"
    ).plot_spectrum2d(show=False)
    plt.savefig(name + "/shimSpectrum2d")

    fig, ax = pyshtools.SHCoeffs.from_array(
        shimmedFieldCoeffs, normalization="ortho"
    ).plot_spectrum(show=False)
    plt.savefig(name + "/shimmedSpectrum")

    fig, ax = pyshtools.SHCoeffs.from_array(
        shimmedFieldCoeffs, normalization="ortho"
    ).plot_spectrum2d(show=False)
    plt.savefig(name + "/shimmedSpectrum2d")

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="polar")  # polar plots
    ax2 = fig.add_subplot(132, projection="polar")
    ax3 = fig.add_subplot(133, projection="polar")

    vec = np.zeros(np.shape(results["bestVector"])[0])
    vec[0] = 1
    shim = np.matmul(shimFieldCoeffs, vec)
    mag1 = reconstructShimmedField(shim, lat, lon)

    vec = np.zeros(np.shape(results["bestVector"])[0])
    vec[-1] = 1
    shim = np.matmul(shimFieldCoeffs, vec)
    mag2 = reconstructShimmedField(shim, lat, lon)

    vec = np.zeros(np.shape(results["bestVector"])[0])
    vec[int(len(vec) / 2)] = 1
    shim = np.matmul(shimFieldCoeffs, vec)
    mag3 = reconstructShimmedField(shim, lat, lon)

    radarplot1 = ax1.pcolormesh(ph_m, th_m, 1e3 * mag1)
    radarplot2 = ax2.pcolormesh(ph_m, th_m, 1e3 * mag2)
    radarplot3 = ax3.pcolormesh(ph_m, th_m, 1e3 * mag3)

    cbar1 = plt.colorbar(radarplot1, ax=ax1)
    cbar1.set_label("field of 1st magnet in mT")
    cbar2 = plt.colorbar(radarplot2, ax=ax2)
    cbar2.set_label("field of last magnet in mT")
    cbar3 = plt.colorbar(radarplot3, ax=ax3)
    cbar3.set_label("field of center magnet in mT")
    plt.tight_layout()
    plt.savefig(name + "/testCoords")
    plt.close(fig)

    # boxplot
    numberOfFirstLsToSkip = 0
    debug = False

    boxplotData = []

    for l in range(1 + numberOfFirstLsToSkip, lmax + 2):
        boxData = fieldCoeffs.copy()
        for l_ in range(l, lmax + 1):
            for m in range(-l_, l_ + 1):
                boxData.set_coeffs(0, l_, m)
        if debug == True:
            fig, ax = boxData.plot_spectrum2d(show=False)
            plt.savefig(name + "/shimmedSpectrum2d_lmax={}".format(l - 1))
            plt.close()
        boxplotData += [1e6 * np.ravel(boxData.expand(lat=lat, lon=lon) - field)]

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure()
    plt.boxplot(boxplotData)
    plt.plot(
        (numberOfFirstLsToSkip, lmax + 1),
        (0.0025 * 1e6 * np.mean(field), 0.0025 * 1e6 * np.mean(field)),
        "k:",
    )
    plt.plot(
        (numberOfFirstLsToSkip, lmax + 1),
        (-0.0025 * 1e6 * np.mean(field), -0.0025 * 1e6 * np.mean(field)),
        "k:",
    )

    plt.ylabel("magnetic field deviation in µT\n(i think so)")
    plt.xlabel("spherical hamonic degree, not nessesaryliy starting at 0!")
    plt.ylim(-500, 500)
    plt.tight_layout()
    plt.savefig(name + "/reconstructionDeviationL")
    plt.close(fig)

    newMaxL = 5
    fieldCoeffsCropped = fieldCoeffs.copy()
    for l in range(newMaxL, lmax - 1):
        for m in range(-m, m + 1):
            fieldCoeffsCropped.set_coeffs(0, l, m)
    fig, ax = fieldCoeffsCropped.plot_spectrum2d(show=False)
    # plt.savefig(name+'/fieldCoeffsCropped_newMaxL={}'.format(newMaxL))
    plt.close()

    boxplotData = []
    for m in range(1, newMaxL + 1):
        boxData = fieldCoeffsCropped.copy()
        for l in range(m, newMaxL + 1):
            for m_ in range(m, l + 1):
                # print('Setting to zero: l, +-m', l, m_)
                boxData.set_coeffs(0, l, m_)
                boxData.set_coeffs(0, l, -m_)
        if debug == True:
            fig, ax = boxData.plot_spectrum2d(show=False)
            plt.savefig(name + "/spectrum2d_mmax={}".format(m))
            plt.close()
        boxplotData += [1e6 * np.ravel(boxData.expand(lat=lat, lon=lon) - field)]

    figsize = (8.3, 8.3 / 3.5)
    fig = plt.figure()
    plt.boxplot(boxplotData)
    plt.plot(
        (numberOfFirstLsToSkip, lmax + 1),
        (0.0025 * 1e6 * np.mean(field), 0.0025 * 1e6 * np.mean(field)),
        "k:",
    )
    plt.plot(
        (numberOfFirstLsToSkip, lmax + 1),
        (-0.0025 * 1e6 * np.mean(field), -0.0025 * 1e6 * np.mean(field)),
        "k:",
    )

    plt.ylabel("magnetic field deviation in µT\n(i think so)")
    plt.xlabel(
        "spherical harmonic degree m for a fixed l!, not nessesarily starting at 0!"
    )
    plt.ylim(-500, 500)
    plt.tight_layout()
    plt.savefig(name + "/reconstructionDeviationM")
    plt.close(fig)

    def plotShimOrientation(name, part, invert_first=False, invert_second=False):
        fig, axs = plt.subplots(2, 1, figsize=(11.6, 16.5))  # 2.2
        fig_empty, axs_empty = plt.subplots(2, 1, figsize=(11.6, 16.5))  # 2.2
        axs = axs.flatten()
        axs_empty = axs_empty.flatten()
        for ax in axs:
            ax.set_xlim(-105, 105)
            ax.set_ylim(-105, 105)
            ax.set_aspect("equal")
            ax.plot(0, 0, "o")
            ax.tick_params(axis="y", direction="in", pad=-30)
            ax.tick_params(axis="x", direction="in", pad=-15)
            ax.set_xticks([-96, -74, -60, -40, -20, 0, 20, 40, 60, 74, 96])
            ax.set_yticks([-96, -74, -60, -40, -20, 20, 40, 60, 74, 96])
            ax.arrow(0, 0, 0, 30, head_width=3)
            ax.arrow(0, 0, 30, 0, head_width=3)
            arrow_properties = dict(facecolor="black", width=0.5, headwidth=9)
            ax.annotate("z axis", xy=(0, 30), xytext=(2, 20))
            ax.annotate("y axis", xy=(30, 0), xytext=(20, 2))
        if part == 0:
            axs = np.concatenate([axs, axs_empty])
        elif part == 1:
            axs = np.concatenate([axs_empty, axs])

        bestVector = np.load(name + "/results.npy", allow_pickle=True).item()[
            "bestVector"
        ]
        params = loadParameters(name)
        zRange = params["xRange"]
        rings = params["numRings"]
        numRotations = params["numRotations"]
        radii = [params["radius"], params["radius2"]]
        radii = np.multiply(1e3, radii)
        numMagnets = [params["numMagnets"], params["numMagnets2"]]
        kValue = params["kValue"]
        # positioning of the magnets in a circle
        if len(zRange) == 2:
            rings = np.linspace(zRange[0], zRange[1], rings)
        elif rings == len(zRange):
            rings = np.array(zRange)
        else:
            print("No clear definition how to place shims...")
        rotation_elements = np.linspace(0, np.pi, numRotations, endpoint=False)

        count = 0
        for rotation in rotation_elements:
            for row_idx, row in enumerate(rings):
                axs[row_idx].legend(["{} mm".format(1e3 * row)])
                for i, radius in enumerate(radii):
                    angle_elements = np.linspace(
                        -np.pi, np.pi, numMagnets[i], endpoint=False
                    )
                    for angle in angle_elements:
                        position = (row, radius * np.cos(angle), radius * np.sin(angle))
                        angle = kValue * angle + rotation
                        dip_vec = [0, np.sin(angle), -np.cos(angle)]
                        dip_vec = np.multiply(2, dip_vec)
                        if bestVector[count] == 1:
                            axs[row_idx].arrow(
                                position[1],
                                position[2],
                                dip_vec[1],
                                dip_vec[2],
                                head_width=3,
                            )
                        elif bestVector[count] == -1:
                            axs[row_idx].arrow(
                                position[1],
                                position[2],
                                -dip_vec[1],
                                -dip_vec[2],
                                head_width=3,
                            )
                        else:
                            pass
                        if rotation == 0:  # count == 0:
                            axs[row_idx].annotate(
                                count, (position[1], position[2]), fontsize=7
                            )
                        count += 1
        fig_empty.clear()

        if invert_first:
            axs[0 + 2 * part].invert_xaxis()
            axs[0 + 2 * part].text(
                -40,
                10,
                "INVERTED",
                fontsize=40,
                color="gray",
                ha="right",
                va="bottom",
                alpha=0.5,
            )
        if invert_second:
            axs[1 + 2 * part].invert_xaxis()
            axs[1 + 2 * part].text(
                -40,
                10,
                "INVERTED",
                fontsize=40,
                color="gray",
                ha="right",
                va="bottom",
                alpha=0.5,
            )
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return fig, axs

    fig, axs = plotShimOrientation(name, part=0, invert_first=False)
    fig.savefig(name + "/orientations_1.pdf")  # , dpi=300)
    fig, axs = plotShimOrientation(name, part=0, invert_first=True, invert_second=True)
    fig.savefig(name + "/orientations_1_inv.pdf")  # , dpi=300)
    fig, axs = plotShimOrientation(name, part=1, invert_first=False)
    fig.savefig(name + "/orientations_2.pdf")  # , dpi=300)
    fig, axs = plotShimOrientation(name, part=1, invert_first=True, invert_second=True)
    fig.savefig(name + "/orientations_2_inv.pdf")
