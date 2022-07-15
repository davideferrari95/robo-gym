#!usr/bin/env/python3

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from math import *

# Create 5 Degrees Polynomial Curve with Period 0-100s
def polynomial_5_deg(T=100):
        
    # Boundaries Conditions
    f_0   = 0.0 # posizione iniziale
    df_0  = 0.0 # velocità iniziale
    ddf_0 = 0.0 # accelerazione iniziale
    f_T   = 1.0 # posizione finale
    df_T  = 0.0 # velocità finale
    ddf_T = 0.0 # accelerazione finale

    a_0 = f_0
    a_1 = df_0
    a_2 = ddf_0 / 2.0
    a_3 = - (20 * f_0 - 20 * f_T + 3 * ddf_0 * pow(T,2) - ddf_T * pow(T,2) + 12 * T * df_0 + 8 * T * df_T) / (2 * pow(T,3))
    a_4 = - (-30 * f_0 + 30 * f_T - 16 * df_0 * T - 14 * df_T * T - 3 * ddf_0 * pow(T,2) + 2 * ddf_T * pow(T,2)) / (2 * pow(T,4))
    a_5 = - (12 * f_0 - 12 * f_T + 6 * df_0 * T + 6 * df_T * T + ddf_0 * pow(T,2) - ddf_T * pow(T,2)) / (2 * pow(T,5))

    # Numero di punti per la stampa della traiettoria
    p = np.linspace(0, T, 500)
    
    # Posizione, Velocità, Accelerazione, Jerk
    f    = a_0 + a_1 * p + a_2 * p**2 + a_3 * p**3 + a_4 * p**4 + a_5 * p**5
    df   = a_1 + 2 * a_2 * p + 3 * a_3 * p**2 + 4 * a_4 * p**3 + 5 * a_5 * p**4
    ddf  = 2 * a_2 + 6 * a_3 * p + 12 * a_4 * p**2 + 20 * a_5 * p**3
    dddf = 6 * a_3 + 24 * a_4 * p + 60 * a_5 * p**2
    
    # Return Splines
    return InterpolatedUnivariateSpline(p, f), InterpolatedUnivariateSpline(p, df), InterpolatedUnivariateSpline(p, ddf), InterpolatedUnivariateSpline(p, dddf)

# Difference Between 2 Angles in Degrees
def angular_difference_degrees(α, β):
    
    # This is either the distance or 360 - distance
    ang = fabs(β - α) % 360
    return 360 - ang if ang > 180 else ang

# Difference Between 2 Angles in Radians
def angular_difference_radians(α, β):
    
    # This is either the distance or 2pi - distance
    ang = fabs(β - α) % (2*pi)
    return 2*pi - ang if ang > pi else ang
