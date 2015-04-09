#!/usr/bin/env python
# encoding: utf-8
"""
airfoilprep.py

Created by Andrew Ning on 2012-04-16.
Copyright (c) NREL. All rights reserved.

Modified by Malo Rosemeier on 2014-05-08


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from math import pi, sin, cos, tan, radians, degrees
import numpy as np
import copy
import nature
from numpy import interp, sqrt, diff
from scipy.integrate import quad
# from scipy.interpolate import RectBivariateSpline



class Polar:
    """
    Defines section lift, drag, and pitching moment coefficients as a
    function of angle of attack at a particular Reynolds number.

    """

    def __init__(self, Re, alpha, cl, cd, cm=None, cr=None):
        """Constructor

        Parameters
        ----------
        Re : float
            Reynolds number
        alpha : ndarray (deg)
            angle of attack
        cl : ndarray
            lift coefficient
        cd : ndarray
            drag coefficient
        cm: ndarray, optional
            pitch coefficient
        cr: float, optional
            local solidity

        """

        self.Re = Re
        self.alpha = np.array(alpha)
        self.cl = np.array(cl)
        self.cd = np.array(cd)
        self.cm = np.zeros_like(cl)
        self.useCM = False
        self.cr = cr
        if cm != None:
            self.useCM = True
            self.cm = cm




    def blend(self, other, weight):
        """Blend this polar with another one with the specified weighting

        Parameters
        ----------
        other : Polar
            another Polar object to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        polar : Polar
            a blended Polar

        """

        # generate merged set of angles of attack - get unique values
        alpha = np.union1d(self.alpha, other.alpha)

        # truncate (TODO: could also have option to just use one of the polars for values out of range)
        min_alpha = max(self.alpha.min(), other.alpha.min())
        max_alpha = min(self.alpha.max(), other.alpha.max())
        alpha = alpha[np.logical_and(alpha >= min_alpha, alpha <= max_alpha)]
        # alpha = np.array([a for a in alpha if a >= min_alpha and a <= max_alpha])

        # interpolate to new alpha
        cl1 = np.interp(alpha, self.alpha, self.cl)
        cl2 = np.interp(alpha, other.alpha, other.cl)
        cd1 = np.interp(alpha, self.alpha, self.cd)
        cd2 = np.interp(alpha, other.alpha, other.cd)

        # linearly blend
        Re = self.Re + weight * (other.Re - self.Re)
        cl = cl1 + weight * (cl2 - cl1)
        cd = cd1 + weight * (cd2 - cd1)

        return Polar(Re, alpha, cl, cd, cm=None, cr=None)



    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """Applies 3-D corrections for rotating sections from the 2-D data.

        Parameters
        ----------
        r_over_R : float
            local radial position / rotor radius
        chord_over_r : float
            local chord length / local radial location
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        polar : Polar
            A new Polar object corrected for 3-D effects

        Notes
        -----
        The Du-Selig method :cite:`Du1998A-3-D-stall-del` is used to correct lift, and
        the Eggers method :cite:`Eggers-Jr2003An-assessment-o` is used to correct drag.


        """

        # rename and convert units for convenience
        alpha = np.radians(self.alpha)
        cl_2d = self.cl
        cd_2d = self.cd
        alpha_max_corr = radians(alpha_max_corr)
        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        # parameters in Du-Selig model
        a = 1
        b = 1
        d = 1
        lam = tsr / (1 + tsr ** 2) ** 0.5  # modified tip speed ratio
        expon = d / lam / r_over_R

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)
        p = np.polyfit(alpha[idx], cl_2d[idx], 1)
        m = p[0]
        alpha0 = -p[1] / m  # AOA with zero lift

        # correction factor
        fcl = 1.0 / m * (1.6 * chord_over_r / 0.1267 * (a - chord_over_r ** expon) / (b + chord_over_r ** expon) - 1)

        # not sure where this adjustment comes from (besides AirfoilPrep spreadsheet of course)
        adj = ((pi / 2 - alpha) / (pi / 2 - alpha_max_corr)) ** 2
        adj[alpha <= alpha_max_corr] = 1.0

        # Du-Selig correction for lift
        cl_linear = m * (alpha - alpha0)
        cl_3d = cl_2d + fcl * (cl_linear - cl_2d) * adj

        # Eggers 2003 correction for drag
        delta_cl3d2d = cl_3d - cl_2d

        delta_cd = delta_cl3d2d * (np.sin(alpha) - 0.12 * np.cos(alpha)) / (np.cos(alpha) + 0.12 * np.sin(alpha))
        cd_3d = cd_2d + delta_cd

        return Polar(self.Re, np.degrees(alpha), cl_3d, cd_3d, cm=None, cr=None)


    def correction3DallModels(self, model, r_over_R, chord_over_r, tsr, R, T, omega, twist,
                alpha_max_corr=30, alpha_linear_min=-5, alpha_linear_max=5,alpha_f1=25,alpha_f0=5, xcoords=None, ycoords=None):
        """Applies 3-D corrections for rotating sections from the 2-D data. for
           different models:
           (1) Snel (for cl) + Eggers (for cd)
           (2) Lindenburg (for cl) + Eggers (for cd)
           (3) Du and Selig
           (4) Chaviaropoulus and Hansen
           (5) Bak

        Author: Malo Rosemeier

        Parameters
        ----------
        model : int
            3D correction model number as shown above
        r_over_R : float
            local radial position / rotor radius
        chord_over_r : float
            local chord length / local radial location
        tsr : float
            tip-speed ratio
        R : float
            rotor radius
        T : float
            operating temperature
        omega : float
            angular velocity
        twist : float
            twist angle
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends
        alpha_f1 : float, optional (deg)
            angle of attack where flow around the airfoil is just about to separate
        alpha_f0 : float, optional (deg)
            angle of attack at which the flow over the airfoil is fully seperated

        Returns
        -------
        polar : Polar
            A new Polar object corrected for 3-D effects

        Notes
        -----


        """

        # rename and convert units to radians for convenience
        alpha = np.radians(self.alpha)
        cl_2d = self.cl
        cd_2d = self.cd
        cm_2d = self.cm
        twist_angle = radians(twist)
        alpha_max_corr = radians(alpha_max_corr)
        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)
        p = np.polyfit(alpha[idx], cl_2d[idx], 1)
        m = p[0]
        alpha0 = -p[1] / m  # AOA with zero lift

        # TODO Bak

        # 3d corrected lift and drag
        cl_linear = m * (alpha - alpha0)
        delta_cl = cl_linear - cl_2d

        # calculate cl_3d
        if model == 1:  # Snel
            fcl = 3 * chord_over_r ** 2
            cl_3d = cl_2d + fcl * delta_cl
        elif model == 2:  # Lindenburg
            r = r_over_R * R
            omega_r = omega * r
            chord = chord_over_r * r
            v_rel = self.Re * nature.air_dynvisc(T) / chord
            fcl = 3.1 * (omega_r / v_rel) ** 2 * chord_over_r ** 2
            cl_3d = cl_2d + fcl * delta_cl
        elif model == 3:  # Du and Selig
            a_3 = 1
            b = 1
            d = 1
            lam = tsr / (1 + tsr ** 2) ** 0.5  # modified tip speed ratio
            expon_cl = d / lam / r_over_R
            expon_cd = expon_cl * 0.5
            fcl = 1.0 / (2 * pi) * (1.6 * chord_over_r / 0.1267 * (a_3 - chord_over_r ** expon_cl) / (b + chord_over_r ** expon_cl) - 1)
            fcd = 1.0 / (2 * pi) * (1.6 * chord_over_r / 0.1267 * (a_3 - chord_over_r ** expon_cd) / (b + chord_over_r ** expon_cd) - 1)
            cl_3d = cl_2d + fcl * delta_cl
        elif model == 4:  # Chaviaropoulus and Hansen
            a_4 = 2.2
            h = 1
            n = 4
            fcl = a_4 * chord_over_r ** h * cos(twist_angle) ** n
            fcd = fcl
            cl_3d = cl_2d + fcl * delta_cl
        elif model == 5:  # Bak
# TODO (Bak model still buggy)
            def __find_index(array, value):
                """ Find closest index of next largest value in an array
                Author: Malo Rosemeier
                """
                return (np.abs(array - value)).argmin()
            # delta cp function
            def __delta_cp(alpha, x_over_c, chord_over_r, r_over_R, twist_angle, alpha_f1, alpha_f0):

                delta_cp_max = 5.0 / 2.0 * \
                    ((1.0 + ((1.0 / r_over_R) ** 2)) ** (1.0 / 2.0)) * chord_over_r / \
                    (1.0 + ((np.tan(alpha + twist_angle)) ** 2))
                    
                delta_cp_var = delta_cp_max * (1.0 - x_over_c) ** 2 * (((alpha - np.radians(alpha_f1)) / \
                    (np.radians(alpha_f0) - np.radians(alpha_f1))) ** 2)  
                                
                delta_cp = []
                for cp_var in delta_cp_var:
                    if cp_var >= delta_cp_max:
                        delta_cp.append(delta_cp_max)
                    else:
                        delta_cp.append(cp_var)
                    
                return delta_cp
            # find index of closest nose coordinate (Leading edge) to get SS coordinates of the airfoil
            idx_le = __find_index(xcoords, 0)
            # calculate slope angles from SS coordinates and one before
            delta_x = diff(np.r_[0.0,xcoords[idx_le:]])
            delta_y = diff(np.r_[0.0,ycoords[idx_le:]])
            slope_angle = np.arctan(delta_y / delta_x)
            
            
            
            # calculate the mean slopes
            #mean_slope_angle =[]
            #for idx in range(len(slope_angle)-1):
            #    mean_slope_angle.append(0.5 * (slope_angle[idx+1]+slope_angle[idx]))
            # add the last slope to the end of the vector to meet required length
            #mean_slope_angle.append(mean_slope_angle[-1])
            #print(mean_slope_angle)
            #print(slope_angle)
            # calculate delta_cp at xcoords
            #print(xcoords[0])
            # loop over alpha array
            delta_cn =[]
            delta_ct =[]
            for angle in alpha:
                normal =[]
                tangential = []
                # calculate pressure delta for one angle
                # TODO: check if twist angle is with correct sign
                abs_delta_cp = __delta_cp(angle, xcoords[idx_le:]-(delta_x/2.0), chord_over_r, r_over_R, twist_angle, alpha_f1, alpha_f0)
                # determine normal part of the pressure
                normal = delta_x * abs_delta_cp #np.cos(slope_angle)
                # determine tangential part
                tangential = -delta_y * abs_delta_cp #np.sin(slope_angle)
                # intgegrate normal
                delta_cn.append(np.sum(normal))
                delta_ct.append(np.sum(tangential))
                
                import matplotlib.pyplot as plt
                #plot = plt.figure
                #p100 = plt.plot(xcoords[idx_le:]-(delta_x/2.0), abs_delta_cp,'r-+') 
                #plt.show()

            
            delta_cl = delta_cn * np.cos(alpha) + delta_ct * np.sin(alpha)
            delta_cd = delta_cn* np.sin(alpha) - delta_ct * np.cos(alpha)
            
            # transform 2d lift and drag into 2d normal and tangential
            ct_2d = np.cos(alpha) * cd_2d - np.sin(alpha) * cl_2d
            cn_2d = np.sin(alpha) * cd_2d + np.cos(alpha) * cl_2d
            
            
            # apply 3d correction deltas
            cn_3d = cn_2d + delta_cn
            ct_3d = ct_2d + delta_ct
            plot = plt.figure
            p99 = plt.plot(alpha, delta_cl,'r-+') 
            p98 = plt.plot(alpha, delta_cd,'g-+')
            p91 = plt.plot(alpha, delta_cn,'y-o') 
            p92 = plt.plot(alpha, delta_ct,'b-o')
            p93 = plt.plot(alpha, cn_2d,'y-*') 
            p94 = plt.plot(alpha, ct_2d,'b-*')
            plt.show()
            #cm_3d = cm_2d + delta_cm
            #cl_3d = cn_3d * np.cos(alpha) + ct_3d * np.sin(alpha)
            cl_3d = cl_2d + delta_cl
            
            plot = plt.figure
            p1 = plt.plot(alpha, cl_3d,'r-o') 
            p10 = plt.plot(alpha, cl_2d,'y-o')
        # calculate cd_3d
        if model == 1:  # Snel or Lindenburg
            # Eggers 2003 correction for drag
            delta_cl3d2d = cl_3d - cl_2d
            delta_cd = delta_cl3d2d * (np.sin(alpha) - 0.12 * np.cos(alpha)) / (np.cos(alpha) + 0.12 * np.sin(alpha))
            cd_3d = cd_2d + delta_cd
        elif model == 2:  # Snel or Lindenburg
            # Eggers 2003 correction for drag
            delta_cl3d2d = cl_3d - cl_2d
            delta_cd = delta_cl3d2d * (np.sin(alpha) - 0.12 * np.cos(alpha)) / (np.cos(alpha) + 0.12 * np.sin(alpha))
            cd_3d = cd_2d + delta_cd
        elif model == 3:  # Du and Selig, Chaviaropoulus and Hansen
            cd_0 = interp(0, alpha, cd_2d, 0, 0)  # drag at alpha=0
            delta_cd = cd_0 - cd_2d
            cd_3d = cd_2d + fcd * delta_cd
        elif model == 4:  # Du and Selig, Chaviaropoulus and Hansen
            cd_0 = interp(0, alpha, cd_2d, 0, 0)  # drag at alpha=0
            delta_cd = cd_0 - cd_2d
            cd_3d = cd_2d + fcd * delta_cd
        elif model == 5:  # Bak
            #cd_3d = cn_3d * np.sin(alpha) - ct_3d * np.cos(alpha)
            cd_3d = cd_2d + delta_cd
            p2 = plt.plot(alpha, cd_3d,'b-o',)
            p20 = plt.plot(alpha, cd_2d,'m-o',)
            plt.grid(True)
            plt.show()
            #plot
        #plot = plt.figure
       
        
        
        
        # calculate cm_3d
        #if model != 5:  # Bak
        cm_3d = self.cm

        return Polar(self.Re, np.degrees(alpha), cl_3d, cd_3d, cm_3d, cr=self.cr)

    def extrapolate(self, cdmax, AR=None, cdmin=0.001, nalpha=15):
        """Extrapolates force coefficients up to +/- 180 degrees using Viterna's method
        :cite:`Viterna1982Theoretical-and`.

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            aspect ratio = (rotor radius / chord_75% radius)
            if provided, cdmax is computed from AR
        cdmin: float, optional
            minimum drag coefficient.  used to prevent negative values that can sometimes occur
            with this extrapolation method
        nalpha: int, optional
            number of points to add in each segment of Viterna method

        Returns
        -------
        polar : Polar
            a new Polar object

        Notes
        -----
        If the current polar already supplies data beyond 90 degrees then
        this method cannot be used in its current form and will just return itself.

        If AR is provided, then the maximum drag coefficient is estimated as

        >>> cdmax = 1.11 + 0.018*AR


        """

        if cdmin < 0:
            raise Exception('cdmin cannot be < 0')

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

        # estimate CD max
        if AR is not None:
            cdmax = 1.11 + 0.018 * AR
        self.cdmax = max(max(self.cd), cdmax)

        # extract matching info from ends
        alpha_high = radians(self.alpha[-1])
        cl_high = self.cl[-1]
        cd_high = self.cd[-1]


        alpha_low = radians(self.alpha[0])
        cl_low = self.cl[0]
        cd_low = self.cd[0]


        if alpha_high > pi / 2:
            raise Exception('alpha[-1] > pi/2')
            return self
        if alpha_low < -pi / 2:
            raise Exception('alpha[0] < -pi/2')
            return self

        # parameters used in model
        sa = sin(alpha_high)
        ca = cos(alpha_high)
        self.A = (cl_high - self.cdmax * sa * ca) * sa / ca ** 2
        self.B = (cd_high - self.cdmax * sa * sa) / ca

        # get pitching moment coefficients (if needed)
        if self.useCM == True:
            def __find_index(array, value):
                """ Find next index of next largest value in an array
                Author: Malo Rosemeier
                """
                idx = (np.abs(array - value)).argmin()
                return idx
            # call CMCoeff
            idx = __find_index(self.alpha, np.degrees(alpha_high))  # find(Alpha1 >= alpha_high,1);
            CLHi = self.cl[idx]
            CDHi = self.cd[idx]
            CMHi = self.cm[idx]

            FoundZeroLift = False;
            # get CM at angle of zero lift (CM0)
            for ii in enumerate(self.alpha):
                if (abs(ii[1]) < 20 and self.cl[ii[0]] <= 0 and self.cl[ii[0] + 1] >= 0):
                    p = -self.cl[ii[0]] / (self.cl[ii[0] + 1] - self.cl[ii[0]])
                    CM0 = self.cm[ii[0]] + p * (self.cm[ii[0] + 1] - self.cm[ii[0]])
                    FoundZeroLift = True
                    break

            if FoundZeroLift != True:  # zero lift not in range of orig table, use first two points
                p = -self.cl[0] / (self.cl[1] - self.cl[0]);
                CM0 = self.cm[0] + p * (self.cm[1] - self.cm[0]);

            XM = (-CMHi + CM0) / (CLHi * cos(alpha_high) + CDHi * sin(alpha_high));
            CMCoef = (XM - 0.25) / tan(alpha_high - pi / 2);

        # alpha_high <-> 90
        alpha1 = np.linspace(alpha_high, pi / 2, nalpha)
        alpha1 = alpha1[1:]  # remove first element so as not to duplicate when concatenating
        cl1, cd1 = self.__Viterna(alpha1, 1.0)

        # 90 <-> 180-alpha_high
        alpha2 = np.linspace(pi / 2, pi - alpha_high, nalpha)
        alpha2 = alpha2[1:]
        cl2, cd2 = self.__Viterna(pi - alpha2, -cl_adj)

        # 180-alpha_high <-> 180
        alpha3 = np.linspace(pi - alpha_high, pi, nalpha)
        alpha3 = alpha3[1:]
        cl3, cd3 = self.__Viterna(pi - alpha3, 1.0)
        cl3 = (alpha3 - pi) / alpha_high * cl_high * cl_adj  # override with linear variation

        # set help variables
        alpha_concat = self.alpha
        cl_concat = self.cl
        cd_concat = self.cd

        if alpha_low <= -alpha_high:
            # alpha4 = []
            # cl4 = []
            # cd4 = []
            # alpha5max = alpha_low
            # Note: modified as it is done in Airfoilprep; alpha_low <-> -alhpa_hi will be overwritten
            index_minalpha_high = __find_index(self.alpha, np.degrees(-alpha_high))
            alpha4 = np.radians(self.alpha[:index_minalpha_high + 1])
            print(np.degrees(alpha4))
            # alpha4 = alpha4[1:-1]  # also remove last element for concatenation for this case
            cl4, cd4 = self.__Viterna(-alpha4, -cl_adj)
            alpha5max = alpha_low

            # remove self.alpha between alpha_low and -alpha_high
            alpha_concat = self.alpha[index_minalpha_high + 1:]
            cl_concat = self.cl[index_minalpha_high + 1:]
            cd_concat = self.cd[index_minalpha_high + 1:]

        else:
            # -alpha_high <-> alpha_low
            # Note: this is done slightly differently than AirfoilPrep for better continuity
            alpha4 = np.linspace(-alpha_high, alpha_low, nalpha)
            alpha4 = alpha4[1:-2]  # also remove last element for concatenation for this case
            cl4 = -cl_high * cl_adj + (alpha4 + alpha_high) / (alpha_low + alpha_high) * (cl_low + cl_high * cl_adj)
            cd4 = cd_low + (alpha4 - alpha_low) / (-alpha_high - alpha_low) * (cd_high - cd_low)
            alpha5max = -alpha_high

        # -90 <-> -alpha_high
        alpha5 = np.linspace(-pi / 2, alpha5max, nalpha)
        alpha5 = alpha5[1:-1]
        cl5, cd5 = self.__Viterna(-alpha5, -cl_adj)

        # -180+alpha_high <-> -90
        alpha6 = np.linspace(-pi + alpha_high, -pi / 2, nalpha)
        alpha6 = alpha6[1:]
        cl6, cd6 = self.__Viterna(alpha6 + pi, cl_adj)

        # -180 <-> -180 + alpha_high
        alpha7 = np.linspace(-pi, -pi + alpha_high, nalpha)
        cl7, cd7 = self.__Viterna(alpha7 + pi, 1.0)
        cl7 = (alpha7 + pi) / alpha_high * cl_high * cl_adj  # linear variation

        alpha = np.concatenate((alpha7, alpha6, alpha5, alpha4, np.radians(alpha_concat), alpha1, alpha2, alpha3))
        cl = np.concatenate((cl7, cl6, cl5, cl4, cl_concat, cl1, cl2, cl3))
        cd = np.concatenate((cd7, cd6, cd5, cd4, cd_concat, cd1, cd2, cd3))

        cd = np.maximum(cd, cdmin)  # don't allow negative drag coefficients
        if alpha5max == -alpha_high:
            cm765 = np.zeros_like(np.concatenate((alpha7, alpha6, alpha5, alpha4)))  # exclude alpha4
        else:
            cm765 = np.zeros_like(np.concatenate((alpha7, alpha6, alpha5)))
        cm123 = np.zeros_like(np.concatenate((alpha1, alpha2, alpha3)))
        cm = np.concatenate((cm765, self.cm, cm123))
        # print(len(alpha4))
        # print(len(alpha))
        # print(len(cm))
        # Do CM calculations and write value to output table
        if self.useCM:

            alphabound = np.r_[-180, -175, -170, -165, 165, 170, 175, 180]
            cmbound = np.r_[0, 0.2, 0.4, 0.35, -0.4, -0.5, -0.25, 0]
            for mm in enumerate(alpha):
                Alfa = np.degrees(mm[1])
                if (Alfa >= np.degrees(alpha_low) and Alfa <= np.degrees(alpha_high)):
                    continue  # no action needed
                if (Alfa > -165 and Alfa < 165):
                    if (abs(Alfa) < 0.01):
                        cm[mm[0]] = CM0
                    else:
                        if Alfa > 0:
                            x = CMCoef * tan(mm[1] - pi / 2) + 0.25
                            cm[mm[0]] = CM0 - x * (cl[mm[0]] * cos(mm[1]) + cd[mm[0]] * sin(mm[1]))
                        else:
                            x = CMCoef * tan(-mm[1] - pi / 2) + 0.25
                            cm[mm[0]] = -(CM0 - x * (-cl[mm[0]] * cos(-mm[1]) + cd[mm[0]] * sin(-mm[1])))
                    continue

                if ((Alfa <= 180 and Alfa >= 165) or (Alfa >= -180 and Alfa <= -165)):
                    cm[mm[0]] = np.interp(Alfa, alphabound, cmbound)

                else:
                    raise Exception('Angle encountered for which there is no CM table value (near +/-180deg).  Program will stop')

        return Polar(self.Re, np.degrees(alpha), cl, cd, cm, cr=self.cr)



    def __Viterna(self, alpha, cl_adj):
        """private method to perform Viterna extrapolation"""

        alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

        cl = self.cdmax / 2 * np.sin(2 * alpha) + self.A * np.cos(alpha) ** 2 / np.sin(alpha)
        cl = cl * cl_adj

        cd = self.cdmax * np.sin(alpha) ** 2 + self.B * np.cos(alpha)

        return cl, cd


    def unsteadyparam(self, alpha_linear_min=-5, alpha_linear_max=5):
        """compute unsteady aero parameters used in AeroDyn input file

        Parameters
        ----------
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        aerodynParam : tuple of floats
            (control setting, stall angle, alpha for 0 cn, cn slope,
            cn at stall+, cn at stall-, alpha for min CD, min(CD))

        """

        alpha = np.radians(self.alpha)
        cl = self.cl
        cd = self.cd

        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        cn = cl * np.cos(alpha) + cd * np.sin(alpha)

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)

        # checks for inppropriate data (like cylinders)
        if len(idx) < 10 or len(np.unique(cl)) < 10:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        # linear fit
        p = np.polyfit(alpha[idx], cn[idx], 1)
        m = p[0]
        alpha0 = -p[1] / m

        # find cn at stall locations
        alphaUpper = np.radians(np.arange(40.0))
        alphaLower = np.radians(np.arange(5.0, -40.0, -1))
        cnUpper = np.interp(alphaUpper, alpha, cn)
        cnLower = np.interp(alphaLower, alpha, cn)
        cnLinearUpper = m * (alphaUpper - alpha0)
        cnLinearLower = m * (alphaLower - alpha0)
        deviation = 0.05  # threshold for cl in detecting stall

        alphaU = np.interp(deviation, cnLinearUpper - cnUpper, alphaUpper)
        alphaL = np.interp(deviation, cnLower - cnLinearLower, alphaLower)

        # compute cn at stall according to linear fit
        cnStallUpper = m * (alphaU - alpha0)
        cnStallLower = m * (alphaL - alpha0)

        # find min cd
        minIdx = cd.argmin()

        # return: control setting, stall angle, alpha for 0 cn, cn slope,
        #         cn at stall+, cn at stall-, alpha for min CD, min(CD)
        return (0.0, degrees(alphaU), degrees(alpha0), m,
                cnStallUpper, cnStallLower, alpha[minIdx], cd[minIdx])




class Airfoil:
    """A collection of Polar objects at different Reynolds numbers

    """

    def __init__(self, polars, name='airfoil', tc=None, r=None, cr=None, xCords=None, yCords=None):
        """Constructor

        Parameters
        ----------
        polars : list(Polar)
            list of Polar objects
        name : string, optional
            airfoil name
        tc : double, optional
            relative thickness
        r : double, optional
            radius
        cr : double, optional
            local solidity (c/r)
        xCords : double, optional
            x-coordinates of the airfoil (starting at TESS, over LE, and ending at TEPS)
        yCords : double, optional
            y-coordinates of the airfoil (starting at TESS, over LE, and ending at TEPS)
        """

        # sort by Reynolds number
        self.polars = polars  # sorted(polars, key=lambda p: p.Re)
        self.name = name
        self.tc = tc
        self.r = r
        self.cr = cr
        self.xCords = xCords
        self.yCords = yCords



    @classmethod
    def initFromAerodynFile(cls, aerodynFile):
        """Construct Airfoil object from AeroDyn file

        Parameters
        ----------
        aerodynFile : str
            path/name of a properly formatted Aerodyn file

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        # open aerodyn file
        f = open(aerodynFile, 'r')

        # skip through header
        f.readline()
        description = f.readline().rstrip()  # remove newline
        f.readline()
        numTables = int(f.readline().split()[0])

        # loop through tables
        for i in range(numTables):

            # read Reynolds number
            Re = float(f.readline().split()[0]) * 1e6

            # read Aerodyn parameters
            param = [0] * 8
            for j in range(8):
                param[j] = float(f.readline().split()[0])

            alpha = []
            cl = []
            cd = []
            # read polar information line by line
            while True:
                line = f.readline()
                if 'EOT' in line:
                    break
                data = [float(s) for s in line.split()]
                alpha.append(data[0])
                cl.append(data[1])
                cd.append(data[2])


            polars.append(Polar(Re, alpha, cl, cd, cm=None))

        f.close()

        return cls(polars)



    def getPolar(self, Re):
        """Gets a Polar object for this airfoil at the specified Reynolds number.

        Parameters
        ----------
        Re : float
            Reynolds number

        Returns
        -------
        obj : Polar
            a Polar object

        Notes
        -----
        Interpolates as necessary. If Reynolds number is larger than or smaller than
        the stored Polars, it returns the Polar with the closest Reynolds number.

        """

        p = self.polars

        if Re <= p[0].Re:
            return copy.deepcopy(p[0])

        elif Re >= p[-1].Re:
            return copy.deepcopy(p[-1])

        else:
            Relist = [pp.Re for pp in p]
            i = np.searchsorted(Relist, Re)
            weight = (Re - Relist[i - 1]) / (Relist[i] - Relist[i - 1])
            return p[i - 1].blend(p[i], weight)



    def blend(self, other, weight):
        """Blend this Airfoil with another one with the specified weighting.


        Parameters
        ----------
        other : Airfoil
            other airfoil to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        obj : Airfoil
            a blended Airfoil object

        Notes
        -----
        First finds the unique Reynolds numbers.  Evaluates both sets of polars
        at each of the Reynolds numbers, then blends at each Reynolds number.

        """

        # combine Reynolds numbers
        Relist1 = [p.Re for p in self.polars]
        Relist2 = [p.Re for p in other.polars]
        Relist = np.union1d(Relist1, Relist2)

        # blend polars
        n = len(Relist)
        polars = [0] * n
        for i in range(n):
            p1 = self.getPolar(Relist[i])
            p2 = other.getPolar(Relist[i])
            polars[i] = p1.blend(p2, weight)


        return Airfoil(polars)


    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """apply 3-D rotational corrections to each polar in airfoil

        Parameters
        ----------
        r_over_R : float
            radial position / rotor radius
        chord_over_r : float
            local chord / local radius
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        airfoil : Airfoil
            airfoil with 3-D corrections

        See Also
        --------
        Polar.correction3D : apply 3-D corrections for a Polar

        """

        n = len(self.polars)
        polars = [0] * n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.correction3D(r_over_R, chord_over_r, tsr, alpha_max_corr, alpha_linear_min, alpha_linear_max)

        return Airfoil(polars)


    def extrapolate(self, cdmax, AR=None, cdmin=0.001):
        """apply high alpha extensions to each polar in airfoil

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            blade aspect ratio (rotor radius / chord at 75% radius).  if included
            it is used to estimate cdmax
        cdmin: minimum drag coefficient

        Returns
        -------
        airfoil : Airfoil
            airfoil with +/-180 degree extensions

        See Also
        --------
        Polar.extrapolate : extrapolate a Polar to high angles of attack

        """

        n = len(self.polars)
        polars = [0] * n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.extrapolate(cdmax, AR, cdmin)

        return Airfoil(polars, name=self.name, tc=self.tc, r=self.r, cr=self.cr, xCords=self.xCords, yCords=self.yCords)



    def interpToCommonAlpha(self, alpha=None):
        """Interpolates all polars to a common set of angles of attack

        Parameters
        ----------
        alpha : ndarray, optional
            common set of angles of attack to use.  If None a union of
            all angles of attack in the polars is used.

        """

        if alpha is None:
            # union of angle of attacks
            alpha = []
            for p in self.polars:
                alpha = np.union1d(alpha, p.alpha)

        # interpolate each polar to new alpha
        n = len(self.polars)
        polars = [0] * n
        for idx, p in enumerate(self.polars):
            cl = np.interp(alpha, p.alpha, p.cl)
            cd = np.interp(alpha, p.alpha, p.cd)
            cm = np.interp(alpha, p.alpha, p.cm)
            polars[idx] = Polar(p.Re, alpha, cl, cd, cm)

        return Airfoil(polars)





    def writeToAerodynFile(self, filename):
        """Write the airfoil section data to a file using AeroDyn input file style.

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        # aerodyn and wtperf require common set of angles of attack
        af = self.interpToCommonAlpha()

        f = open(filename, 'w')

        print >> f, 'AeroDyn airfoil file.'
        print >> f, 'Compatible with AeroDyn v13.0.'
        print >> f, 'Generated by airfoilprep.py'
        print >> f, '{0:<10d}\t\t{1:40}'.format(len(af.polars), 'Number of airfoil tables in this file')
        for p in af.polars:
            print >> f, '{0:<10f}\t{1:40}'.format(p.Re / 1e6, 'Reynolds number in millions.')
            param = p.unsteadyparam()
            print >> f, '{0:<10f}\t{1:40}'.format(param[0], 'Control setting')
            print >> f, '{0:<10f}\t{1:40}'.format(param[1], 'Stall angle (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[2], 'Angle of attack for zero Cn for linear Cn curve (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[3], 'Cn slope for zero lift for linear Cn curve (1/rad)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[4], 'Cn at stall value for positive angle of attack for linear Cn curve')
            print >> f, '{0:<10f}\t{1:40}'.format(param[5], 'Cn at stall value for negative angle of attack for linear Cn curve')
            print >> f, '{0:<10f}\t{1:40}'.format(param[6], 'Angle of attack for minimum CD (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[7], 'Minimum CD value')
            for a, cl, cd, cm in zip(p.alpha, p.cl, p.cd, p.cm):
                print >> f, '{:<10f}\t{:<10f}\t{:<10f}\t{:<10f}'.format(a, cl, cd, cm)
            print >> f, 'EOT'
        f.close()








    def createDataGrid(self):
        """interpolate airfoil data onto uniform alpha-Re grid.

        Returns
        -------
        alpha : ndarray (deg)
            a common set of angles of attack (union of all polars)
        Re : ndarray
            all Reynolds numbers defined in the polars
        cl : ndarray
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : ndarray
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """

        af = self.interpToCommonAlpha()
        polarList = af.polars

        # angle of attack is already same for each polar
        alpha = polarList[0].alpha

        # all Reynolds numbers
        Re = [p.Re for p in polarList]

        # fill in cl, cd grid
        cl = np.zeros((len(alpha), len(Re)))
        cd = np.zeros((len(alpha), len(Re)))

        for (idx, p) in enumerate(polarList):
            cl[:, idx] = p.cl
            cd[:, idx] = p.cd

        return alpha, Re, cl, cd




    # def evaluate(self, alpha, Re):
    #     """Get lift/drag coefficient at the specified angle of attack and Reynolds number

    #     Parameters
    #     ----------
    #     alpha : float (rad)
    #         angle of attack (in Radians!)
    #     Re : float
    #         Reynolds number

    #     Returns
    #     -------
    #     cl : float
    #         lift coefficient
    #     cd : float
    #         drag coefficient

    #     Notes
    #     -----
    #     Uses a spline so that output is continuously differentiable
    #     also uses a small amount of smoothing to help remove spurious multiple solutions

    #     """

    #     # setup spline if necessary
    #     if self.need_to_setup_spline:
    #         alpha_v, Re_v, cl_M, cd_M = self.createDataGrid()
    #         alpha_v = np.radians(alpha_v)

    #         # special case if zero or one Reynolds number (need at least two for bivariate spline)
    #         if len(Re_v) < 2:
    #             Re_v = [1e1, 1e15]
    #             cl_M = np.c_[cl_M, cl_M]
    #             cd_M = np.c_[cd_M, cd_M]

    #         kx = min(len(alpha_v)-1, 3)
    #         ky = min(len(Re_v)-1, 3)

    #         self.cl_spline = RectBivariateSpline(alpha_v, Re_v, cl_M, kx=kx, ky=ky, s=0.1)
    #         self.cd_spline = RectBivariateSpline(alpha_v, Re_v, cd_M, kx=kx, ky=ky, s=0.001)
    #         self.need_to_setup_spline = False

    #     # evaluate spline --- index to make scalar

    #     cl = self.cl_spline.ev(alpha, Re)[0]
    #     cd = self.cd_spline.ev(alpha, Re)[0]

    #     return cl, cd



class AirfoilSet:
    """A Set of Collections of Polar objects at different Reynolds numbers

    Author: Malo Rosemeier

    """

    def __init__(self, airfoils):
        """Constructor

        Parameters
        ----------
        airfoils : list(Airfoil)
            list of Airfoil objects

        """

        self.airfoils = airfoils  # , key=lambda p: airfoils.p.tc)

    def writeToHAWC2pcFile(self, filename):
        """Write the airfoil section data to a file using HAWC2 input file style. (pc file)

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        af = self.airfoils

        f = open(filename, 'w')

        # length of the list of airfoils
        number_afs = len(af)
        print(number_afs)
        print >> f, '1 Blade aerodynamic coefficients'
        print >> f, str(number_afs)
        for no, airfoil in enumerate(af):
            print >> f, '{:<3d}  {:<3d} {:.3f} \t {}'.format(no + 1, len(airfoil.polars[0].alpha), airfoil.tc * 100, airfoil.name)
            for a, cl, cd, cm in zip(airfoil.polars[0].alpha, airfoil.polars[0].cl, airfoil.polars[0].cd, airfoil.polars[0].cm):
                print >> f, '{:.2f} \t {:.4e} \t {:.4e} \t {:.4e}'.format(a, cl, cd, cm)
        f.close()

    def writeToHAWC2aeFile(self, filename):
        """Write the airfoil section data to a file using HAWC2 input file style. (ae file)

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        af = self.airfoils

        f = open(filename, 'w')

        # determine hub radius
        hubRadius = af[-1].r

        # length of the list of airfoils
        number_afs = len(af)
        print(number_afs)
        print >> f, '1'
        print >> f, '1 ' + str(number_afs)

        for airfoil in af:
            print >> f, '{:.3f}  {:.3f} {:.3f}'.format(airfoil.r - hubRadius, airfoil.cr / airfoil.r, airfoil.tc * 100)
            # for a, cl, cd, cm in zip(airfoil.polars[0].alpha, airfoil.polars[0].cl, airfoil.polars[0].cd, airfoil.polars[0].cm):
                # print >> f, '{:.2f} \t {:.4e} \t {:.4e} \t {:.4e}'.format(a, cl, cd, cm)
        f.close()




if __name__ == '__main__':

    import os
    from argparse import ArgumentParser, RawTextHelpFormatter

    # setup command line arguments
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                            description='Preprocessing airfoil data for wind turbine applications.')
    parser.add_argument('src_file', type=str, help='source file')
    parser.add_argument('--stall3D', type=str, nargs=3, metavar=('r/R', 'c/r', 'tsr'), help='2D data -> apply 3D corrections')
    parser.add_argument('--extrap', type=str, nargs=1, metavar=('cdmax'), help='3D data -> high alpha extrapolations')
    parser.add_argument('--blend', type=str, nargs=2, metavar=('otherfile', 'weight'), help='blend 2 files weight 0: sourcefile, weight 1: otherfile')
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--plot', action='store_true', help='plot data using matplotlib')
    parser.add_argument('--common', action='store_true', help='interpolate the data at different Reynolds numbers to a common set of angles of attack')


    # parse command line arguments
    args = parser.parse_args()
    fileOut = args.out

    if args.plot:
        import matplotlib.pyplot as plt

    # perform actions
    if args.stall3D is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_3D' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)
        floats = [float(var) for var in args.stall3D]
        af3D = af.correction3D(*floats)

        if args.common:
            af3D = af3D.interpToCommonAlpha()

        af3D.writeToAerodynFile(fileOut)

        if args.plot:

            for p, p3D in zip(af.polars, af3D.polars):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                plt.plot(p.alpha, p.cl, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cl, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend(loc='lower right')

                # plt.subplot(122)
                plt.figure()
                plt.plot(p.alpha, p.cd, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cd, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend(loc='upper center')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/stall3d.pdf')

            plt.show()


    elif args.extrap is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_extrap' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)
        afext = af.extrapolate(float(args.extrap[0]))

        if args.common:
            afext = afext.interpToCommonAlpha()

        afext.writeToAerodynFile(fileOut)

        if args.plot:

            for p, pext in zip(af.polars, afext.polars):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cl, 'r')
                p2, = plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.subplot(122)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cd, 'r')
                p2, = plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='lower right')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/extrap.pdf')

            plt.show()


    elif args.blend is not None:

        if fileOut is None:
            name1, ext = os.path.splitext(args.src_file)
            name2, ext = os.path.splitext(os.path.basename(args.blend[0]))
            fileOut = name1 + '+' + name2 + '_blend' + args.blend[1] + ext

        af1 = Airfoil.initFromAerodynFile(args.src_file)
        af2 = Airfoil.initFromAerodynFile(args.blend[0])
        afOut = af1.blend(af2, float(args.blend[1]))

        if args.common:
            afOut = afOut.interpToCommonAlpha()

        afOut.writeToAerodynFile(fileOut)



        if args.plot:

            for p in afOut.polars:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.text(0.6, 0.2, 'Re = ' + str(p.Re / 1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re / 1e6) + ' million', transform=ax.transAxes)

            plt.show()





