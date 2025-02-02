"""
Created on Thu Jan 23 12:49:25 2025

@author: esoria
@reviewer: cruiz

Class created from esoria code in order to have a modulable code
"""
import numpy as np
from Berreman import Berreman
import Berreman.materials as mat

### CONSTANTS ###
pol_waves = [393.3, 396.8, 422.7, 453, 455.4, 460.7, 492.3, 514, 517.3,
                     525, 587.6, 589, 617.3, 630, 656.3, 709, 815.1, 849.8, 854,
                     1083.0, 1565.0]

ANGLES = [0, 90, 63.75, 153.75, 0, 90]

class Retarder(object):
    def __init__(self):
        self.berreman = Berreman.Berreman()

        #dummy wavelength init:
        self.wavelength = pol_waves[0]

        #crystal orientations:
        self.psi, self.eta, self.xi = 0, 0, 0
        self.eta_angles = self.compute_eta_angles()

        #CTEs quartz
        self.alpha_o, self.alpha_e = 7.1*1e-6, 13.2*1e-6
        #TOC quartz
        self.toc_o, self.toc_e = 6.5*1e-6, -5.5*1e-6

        """refraction index"""
        ## Mgf2 dummy init
        self.no_c, self.ne_c = None, None

        self.n_oil = 1 #Meadowlark index-matching oil
        #Dummy infrasil init
        self.n_window = None
        ## QUARTZ
        self.no, self.ne = None, None

        self.get_refractiveindex(self.wavelength)

        """thicknessess"""
        ###THICKNESS in nm
        self.PLATE_THICKNESS = [2124900.0, 2099100.0, 2132400.0, 2099100.0, 2124900.0, 2099100.0]
        self.WINDOW_THICKNESS = 10000000.0
        self.OIL_THICNKESS = 10000.0
        self.COATING_THICKNESS = 94.4
        ##Optical thicnkess
        self.thkW = None  #window
        self.thkO = None #oil
        self.thkC = None #coating
        self.plate_thickness = None

        self.get_opticalthicnkess(self.wavelength)

        """field matrix in/out air"""
        self.Fc = self.berreman.fmat([1, 1, 1, 0, 0, 0], 0)  # field matrix for air at normal incidence, n=1, beta=0
        self.Fs = self.berreman.fmat([1, 1, 1, 0, 0, 0], 0)  # field matrix for air output interface

    def compute_eta_angles(self):
        """

        :return: clocking angle for each retarder plate
        """
        return [self.eta + np.radians(angle) for angle in ANGLES]

    def get_refractiveindex(self, wl):
        """

        :param wl: wavelength in nanometers
        :return:
        """
        inx2 = mat.mgf2(wl)
        self.no_c, self.ne_c = inx2[0, 0], inx2[0, 1]
        self.n_window = mat.fusedsilica(wl)
        self.no, self.ne = mat.quartz_retarder(wl)

    def get_opticalthicnkess(self, wavelength):
        self.thkW = self.compute_optical_thickness(self.WINDOW_THICKNESS, wavelength) #window
        self.thkO = self.compute_optical_thickness(self.OIL_THICNKESS, wavelength) #oil
        self.thkC = self.compute_optical_thickness(self.COATING_THICKNESS, wavelength) #coating
        self.plate_thickness = self.compute_optical_thickness(self.PLATE_THICKNESS, wavelength)

    def compute_optical_thickness(self, thickness, wavelength):
        if isinstance(thickness, list):
            return [t / wavelength for t in thickness]
        else:
            return thickness/wavelength

    def create_layer_stack(self, bonding=None, window=True):
        layers = []

        if bonding == "oil":
            if window: # case with infrasil in/out window
                layers += [
                    [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Input Coating
                    [self.n_window, self.n_window, self.n_window, 0, 0, 0, self.thkW],  # Input Window
                    [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Coating after Window
                    [self.n_oil, self.n_oil, self.n_oil, 0, 0, 0, self.thkO]  # Oil Layer
                ]
            # Main stack: (Coating → Quartz → Coating → Oil) x 5
            for thk, eta in zip(self.plate_thickness[:-1], self.eta_angles[:-1]):
                layers += [
                    [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Coating
                    [self.no, self.ne, self.no, eta, self.psi, self.xi, thk],  # Quartz
                    [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Coating
                    [self.n_oil, self.n_oil, self.n_oil, 0, 0, 0, self.thkO]  # Oil
                ]
            # Final coating → quartz → coating
            layers += [
                [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Coating
                [self.no, self.ne, self.no, self.eta_angles[-1], self.psi, self.xi, self.plate_thickness[-1]],
                # Final Quartz
                [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC]  # Coating
            ]
            # Add output window if included
            if window:
                layers += [
                    [self.n_oil, self.n_oil, self.n_oil, 0, 0, 0, self.thkO],  # Oil
                    [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Coating
                    [self.n_window, self.n_window, self.n_window, 0, 0, 0, self.thkW],  # Output Window
                    [self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC],  # Output Coating
                ]
        elif bonding == "contact": #No window
            layers += [[self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC]] # top coating
            #Uncoated quartz stack
            for thk, eta in zip(self.plate_thickness, self.eta_angles):
                layers += [[self.no, self.ne, self.no, eta, self.psi, self.xi, thk]]
            layers += [[self.ne_c, self.no_c, self.no_c, 0, 0, 0, self.thkC]]  # bottom coating

        return np.asarray(layers)

    def compute_mueller_stack(self, wavelength, bonding=None, window=True):
        #update refraction index values
        self.get_refractiveindex(wavelength)

        #updated optical thicness
        self.get_opticalthicnkess(wavelength)

        #create layer stack
        layer = self.create_layer_stack(bonding=bonding, window=window)
        #compute mueller matrix elements
        M = self.berreman.cmat(layer, 0.0) # characteristics matrix
        MMr, MMt, Sr, St = self.berreman.cmmat(self.Fc, M, self.Fs)

        #Normalize the transmission matrix
        MMT_00 = MMt[0, 0]
        MMt /= MMT_00
        MMt[0, 0] = MMT_00
        return MMt




