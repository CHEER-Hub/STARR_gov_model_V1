#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:40:48 2024

@author: Jingya Wang
"""

""" Calculate equilibrium prices for competitive market """

import numpy as np
from scipy.optimize import curve_fit
import sympy as sp

import warnings
# Suppress warnings
warnings.filterwarnings('ignore')


#%%
# Polynomial fitting
def poly4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def poly2(x, a, b, c):
    return a*x**2 + b*x + c

#%%
def Func_cal_equilibrium(P_sum_avehurr_duo_fldwd_LH, Q_sum_avehurr_fldwd_LR, Q_sum_avehurr_fldwd_SFHA, FX_duo, lambda_LR, lambda_SFHA, clevel, insurance_price_LB, insurance_price_UB):

    ########
    ##•	Initialize parameters and variables, including (i) equilibrium price variables “Eprice_perlvl_LR” and “Eprice_perlvl_SFHA”, 
    ##and (ii) equilibrium demand variables “Edemand_perlvl_LR” and “Edemand_perlvl_SFHA”
    track_price = []
    Eprice_perlvl_LR = np.zeros(clevel)
    Eprice_perlvl_SFHA = np.zeros(clevel)
    Edemand_perlvl_LR = np.zeros(clevel)
    Edemand_perlvl_SFHA = np.zeros(clevel)
    
    
    
    ########
    ## •	Create a 4th-order polynomial fit for demand and prices in high-risk zones and record corresponding coefficients
    ## •	Create a 2nd-order polynomial fit for demand and prices in low-risk zones and record corresponding coefficients
    ## •	Create a polynomial surface fit for demand and total cost with degree of 1 for the low-risk demand and degree of 2 for the high-risk demand. Record corresponding coefficients
    ## •	Based on literature: Gao, Y., Nozick, L., Kruse, J., & Davidson, R. (2016). Modeling competition in a market for natural catastrophe insurance. Journal of Insurance Issues, 38-68.
    
    
    
    # Fit a 4th degree polynomial for SFHA
    ps_coeffs = np.polyfit(Q_sum_avehurr_fldwd_SFHA, insurance_price_LB + lambda_SFHA, 4)
    ps4, ps3, ps2, ps1, ps0 = ps_coeffs
    
    # Fit a 2nd degree polynomial for LR
    pL_coeffs = np.polyfit(Q_sum_avehurr_fldwd_LR, insurance_price_LB + lambda_LR, 2)
    pL2, pL1, pL0 = pL_coeffs
    
    # Generate meshgrid
    lambda_LR_mesh, lambda_SFHA_mesh = np.meshgrid(insurance_price_LB + lambda_LR, insurance_price_LB + lambda_SFHA)
    
    cost_year = P_sum_avehurr_duo_fldwd_LH - FX_duo

    len_data = len(lambda_LR) * len(lambda_SFHA)
    tempx = np.zeros(len_data)
    tempy = np.zeros(len_data)
    tempz = np.zeros(len_data)
    k = 0
    for i in range(len(lambda_LR)):
        for j in range(len(lambda_SFHA)):
            tempx[k] = Q_sum_avehurr_fldwd_LR[i]
            tempy[k] = Q_sum_avehurr_fldwd_SFHA[j]
            tempz[k] = cost_year[i, j]
            k += 1
    
    # Define a function to fit the data
    def poly12(x, p00, p10, p01, p11, p02):
        return p00 + p10*x[0] + p01*x[1] + p11*x[0]*x[1] + p02*x[1]**2
    
    # Fit the polynomial using curve_fit
    popt, _ = curve_fit(poly12, (tempx, tempy), tempz)
    p00, p10, p01, p11, p02 = popt
    
    # ####sklearn method
    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.linear_model import LinearRegression
    # from sklearn.preprocessing import normalize
    
    # # Combine tempx and tempy into a single 2D array
    # X = np.vstack((tempx, tempy)).T

    # # Create polynomial features for the specific model 'poly12'
    # poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    # X_poly = poly.fit_transform(X)
    
    # # Fit the polynomial model
    # model = LinearRegression()
    # model.fit(X_poly, tempz)
    
    # # Get the coefficients
    # intercept = model.intercept_
    # coefficients = model.coef_
    
    # # Map coefficients to corresponding polynomial terms
    # p00 = intercept          # Corresponds to the constant term
    # p10 = coefficients[0]    # Corresponds to the x term
    # p01 = coefficients[1]    # Corresponds to the y term
    # p11 = coefficients[2]    # Corresponds to the x*y term
    # p02 = coefficients[3]    # Corresponds to the y^2 term
    
    
    
    ########
    ## •	Solve optimal demand and prices for low- and high-risk zones
    ## •	Based on literature: Gao, Y., Nozick, L., Kruse, J., & Davidson, R. (2016). Modeling competition in a market for natural catastrophe insurance. Journal of Insurance Issues, 38-68.
    
    z = sp.symbols('z')
    for n in range(1, clevel + 1): # different markets, 1:4
        
        polynomial = (
            32 * n**8 * pL2 * ps4**2 * z**8 + 32 * n**7 * pL2 * ps4**2 * z**8 + 10 * n**9 * pL2 * ps4**2 * z**8 +
            n**10 * pL2 * ps4**2 * z**8 + 52 * n**7 * pL2 * ps3 * ps4 * z**7 + 48 * n**6 * pL2 * ps3 * ps4 * z**7 +
            18 * n**8 * pL2 * ps3 * ps4 * z**7 + 2 * n**9 * pL2 * ps3 * ps4 * z**7 + 40 * n**6 * pL2 * ps2 * ps4 * z**6 +
            32 * n**5 * pL2 * ps2 * ps4 * z**6 + 16 * n**7 * pL2 * ps2 * ps4 * z**6 + 2 * n**8 * pL2 * ps2 * ps4 * z**6 +
            21 * n**6 * pL2 * ps3**2 * z**6 + 18 * n**5 * pL2 * ps3**2 * z**6 + 8 * n**7 * pL2 * ps3**2 * z**6 +
            n**8 * pL2 * ps3**2 * z**6 + 32 * n**5 * pL2 * ps2 * ps3 * z**5 + 28 * n**5 * pL2 * ps1 * ps4 * z**5 +
            24 * n**4 * pL2 * ps2 * ps3 * z**5 + 14 * n**6 * pL2 * ps2 * ps3 * z**5 + 14 * n**6 * pL2 * ps1 * ps4 * z**5 +
            16 * n**4 * pL2 * ps1 * ps4 * z**5 + 2 * n**7 * pL2 * ps2 * ps3 * z**5 + 2 * n**7 * pL2 * ps1 * ps4 * z**5 -
            32 * n**4 * p02 * pL2 * ps4 * z**5 - 24 * n**5 * p02 * pL2 * ps4 * z**5 - 4 * n**6 * p02 * pL2 * ps4 * z**5 +
            22 * n**4 * pL2 * ps1 * ps3 * z**4 + 16 * n**4 * pL2 * ps0 * ps4 * z**4 + 12 * n**5 * pL2 * ps1 * ps3 * z**4 +
            12 * n**5 * pL2 * ps0 * ps4 * z**4 + 12 * n**3 * pL2 * ps1 * ps3 * z**4 + 2 * n**6 * pL2 * ps1 * ps3 * z**4 +
            2 * n**6 * pL2 * ps0 * ps4 * z**4 - 24 * n**3 * p02 * pL2 * ps3 * z**4 - 20 * n**4 * p02 * pL2 * ps3 * z**4 -
            16 * n**4 * p01 * pL2 * ps4 * z**4 + 5 * n**4 * p11 * pL1 * ps4 * z**4 - 12 * n**5 * p01 * pL2 * ps4 * z**4 +
            4 * n**3 * p11 * pL1 * ps4 * z**4 - 4 * n**5 * p02 * pL2 * ps3 * z**4 - 2 * n**6 * p01 * pL2 * ps4 * z**4 +
            n**5 * p11 * pL1 * ps4 * z**4 + 12 * n**4 * pL2 * ps2**2 * z**4 + 6 * n**5 * pL2 * ps2**2 * z**4 +
            8 * n**3 * pL2 * ps2**2 * z**4 + n**6 * pL2 * ps2**2 * z**4 + 16 * n**3 * pL2 * ps1 * ps2 * z**3 +
            12 * n**3 * pL2 * ps0 * ps3 * z**3 + 10 * n**4 * pL2 * ps1 * ps2 * z**3 + 10 * n**4 * pL2 * ps0 * ps3 * z**3 +
            8 * n**2 * pL2 * ps1 * ps2 * z**3 + 2 * n**5 * pL2 * ps1 * ps2 * z**3 + 2 * n**5 * pL2 * ps0 * ps3 * z**3 -
            16 * n**3 * p02 * pL2 * ps2 * z**3 - 16 * n**2 * p02 * pL2 * ps2 * z**3 + 4 * n**3 * p11 * pL1 * ps3 * z**3 -
            12 * n**3 * p01 * pL2 * ps3 * z**3 - 10 * n**4 * p01 * pL2 * ps3 * z**3 + 3 * n**2 * p11 * pL1 * ps3 * z**3 -
            4 * n**4 * p02 * pL2 * ps2 * z**3 - 2 * n**5 * p01 * pL2 * ps3 * z**3 + n**4 * p11 * pL1 * ps3 * z**3 +
            2 * n * p11 * pL1 * ps2 * z**2 - 8 * n * p02 * pL2 * ps1 * z**2 + 8 * n**3 * pL2 * ps0 * ps2 * z**2 +
            8 * n**2 * pL2 * ps0 * ps2 * z**2 + 2 * n**4 * pL2 * ps0 * ps2 * z**2 + 3 * n**2 * p11 * pL1 * ps2 * z**2 -
            12 * n**2 * p02 * pL2 * ps1 * z**2 - 8 * n**3 * p01 * pL2 * ps2 * z**2 - 8 * n**2 * p01 * pL2 * ps2 * z**2 -
            4 * n**3 * p02 * pL2 * ps1 * z**2 - 2 * n**4 * p01 * pL2 * ps2 * z**2 + 2 * n * pL2 * ps1**2 * z**2 +
            8 * n * p02**2 * pL2 * z**2 + n**3 * p11 * pL1 * ps2 * z**2 + 4 * n**3 * pL2 * ps1**2 * z**2 +
            5 * n**2 * pL2 * ps1**2 * z**2 + 4 * n**2 * p02**2 * pL2 * z**2 + n**4 * pL2 * ps1**2 * z**2 +
            6 * n**2 * pL2 * ps0 * ps1 * z + 2 * n**3 * pL2 * ps0 * ps1 * z - 6 * n**2 * p01 * pL2 * ps1 * z -
            4 * n**2 * p02 * pL2 * ps0 * z - 2 * n**3 * p01 * pL2 * ps1 * z + 4 * n**2 * p01 * p02 * pL2 * z +
            4 * n * pL2 * ps0 * ps1 * z + 2 * n * p11 * pL1 * ps1 * z - 8 * n * p02 * pL2 * ps0 * z -
            4 * n * p01 * pL2 * ps1 * z - 2 * n * p02 * p11 * pL1 * z + 8 * n * p01 * p02 * pL2 * z +
            n**2 * p11 * pL1 * ps1 * z - 2 * p02 * p11 * pL1 * z + p11 * pL1 * ps1 * z - p11**3 * z -
            2 * n**2 * p01 * pL2 * ps0 - 4 * n * p01 * pL2 * ps0 - n * p01 * p11 * pL1 - p01 * p11 * pL1 +
            n * p11 * pL1 * ps0 + 2 * n * pL2 * ps0**2 + 2 * n * p01**2 * pL2 + p11 * pL1 * ps0 - p10 * p11**2 +
            n**2 * pL2 * ps0**2 + n**2 * p01**2 * pL2 + p11**2 * pL0
        )


            
        solz = sp.solve(polynomial, z)
        temp_solz = [complex(root.evalf()) for root in solz]
        
        rec_solz = [i for i, root in enumerate(temp_solz) if abs(root.real) < 10000 * abs(root.imag)]
        for i in sorted(rec_solz, reverse=True):
            temp_solz.pop(i)
        
        y1 = [root.real for root in temp_solz]
        #high risk
        price_s = [ps4*(n*y)**4 + ps3*(n*y)**3 + ps2*(n*y)**2 + ps1*(n*y) + ps0 for y in y1]
        
        x1 = [(ps0 - p01 + (4*ps4*n**3*y**4 + 3*ps3*n**2*y**3 + 2*ps2*n*y**2 + ps1*y) - 2*p02*y + n**4*ps4*y**4 +
               n**3*ps3*y**3 + n**2*ps2*y**2 + n*ps1*y)/p11 for y in y1]
        # low risk
        price_l = [pL2*(n*x)**2 + pL1*(n*x) + pL0 for x in x1]
        
        
        
        # polynomial_eq = 32*n**8*pL2*ps4**2*z**8 + 32*n**7*pL2*ps4**2*z**8 + 10*n**9*pL2*ps4**2*z**8 + \
        #         n**10*pL2*ps4**2*z**8 + 52*n**7*pL2*ps3*ps4*z**7 + 48*n**6*pL2*ps3*ps4*z**7 + \
        #         18*n**8*pL2*ps3*ps4*z**7 + 2*n**9*pL2*ps3*ps4*z**7 + 40*n**6*pL2*ps2*ps4*z**6 + \
        #         32*n**5*pL2*ps2*ps4*z**6 + 16*n**7*pL2*ps2*ps4*z**6 + 2*n**8*pL2*ps2*ps4*z**6 + \
        #         21*n**6*pL2*ps3**2*z**6 + 18*n**5*pL2*ps3**2*z**6 + 8*n**7*pL2*ps3**2*z**6 + \
        #         n**8*pL2*ps3**2*z**6 + 32*n**5*pL2*ps2*ps3*z**5 + 28*n**5*pL2*ps1*ps4*z**5 + \
        #         24*n**4*pL2*ps2*ps3*z**5 + 14*n**6*pL2*ps2*ps3*z**5 + 14*n**6*pL2*ps1*ps4*z**5 + \
        #         16*n**4*pL2*ps1*ps4*z**5 + 2*n**7*pL2*ps2*ps3*z**5 + 2*n**7*pL2*ps1*ps4*z**5 - \
        #         32*n**4*p02*pL2*ps4*z**5 - 24*n**5*p02*pL2*ps4*z**5 - 4*n**6*p02*pL2*ps4*z**5 + \
        #         22*n**4*pL2*ps1*ps3*z**4 + 16*n**4*pL2*ps0*ps4*z**4 + 12*n**5*pL2*ps1*ps3*z**4 + \
        #         12*n**5*pL2*ps0*ps4*z**4 + 12*n**3*pL2*ps1*ps3*z**4 + 2*n**6*pL2*ps1*ps3*z**4 + \
        #         2*n**6*pL2*ps0*ps4*z**4 - 24*n**3*p02*pL2*ps3*z**4 - 20*n**4*p02*pL2*ps3*z**4 - \
        #         16*n**4*p01*pL2*ps4*z**4 + 5*n**4*p11*pL1*ps4*z**4 - 12*n**5*p01*pL2*ps4*z**4 + \
        #         4*n**3*p11*pL1*ps4*z**4 - 4*n**5*p02*pL2*ps3*z**4 - 2*n**6*p01*pL2*ps4*z**4 + \
        #         n**5*p11*pL1*ps4*z**4 + 12*n**4*pL2*ps2**2*z**4 + 6*n**5*pL2*ps2**2*z**4 + \
        #         8*n**3*pL2*ps2**2*z**4 + n**6*pL2*ps2**2*z**4 + 16*n**3*pL2*ps1*ps2*z**3 + \
        #         12*n**3*pL2*ps0*ps3*z**3 + 10*n**4*pL2*ps1*ps2*z**3 + 10*n**4*pL2*ps0*ps3*z**3 + \
        #         8*n**2*pL2*ps1*ps2*z**3 + 2*n**5*pL2*ps1*ps2*z**3 + 2*n**5*pL2*ps0*ps3*z**3 - \
        #         16*n**3*p02*pL2*ps2*z**3 - 16*n**2*p02*pL2*ps2*z**3 + 4*n**3*p11*pL1*ps3*z**3 - \
        #         12*n**3*p01*pL2*ps3*z**3 - 10*n**4*p01*pL2*ps3*z**3 + 3*n**2*p11*pL1*ps3*z**3 - \
        #         4*n**4*p02*pL2*ps2*z**3 - 2*n**5*p01*pL2*ps3*z**3 + n**4*p11*pL1*ps3*z**3 + \
        #         2*n*p11*pL1*ps2*z**2 - 8*n*p02*pL2*ps1*z**2 + 8*n**3*pL2*ps0*ps2*z**2 + \
        #         8*n**2*pL2*ps0*ps2*z**2 + 2*n**4*pL2*ps0*ps2*z**2 + 3*n**2*p11*pL1*ps2*z**2 - \
        #         12*n**2*p02*pL2*ps1*z**2 - 8*n**3*p01*pL2*ps2*z**2 - 8*n**2*p01*pL2*ps2*z**2 - \
        #         4*n**3*p02*pL2*ps1*z**2 - 2*n**4*p01*pL2*ps2*z**2 + 2*n*pL2*ps1**2*z**2 + \
        #         8*n*p02**2*pL2*z**2 + n**3*p11*pL1*ps2*z**2 + 4*n**3*pL2*ps1**2*z**2 + \
        #         5*n**2*pL2*ps1**2*z**2 + 4*n**2*p02**2*pL2*z**2 + n**4*pL2*ps1**2*z**2 + \
        #         6*n**2*pL2*ps0*ps1*z + 2*n**3*pL2*ps0*ps1*z - 6*n**2*p01*pL2*ps1*z - \
        #         4*n**2*p02*pL2*ps0*z - 2*n**3*p01*pL2*ps1*z + 4*n**2*p01*p02*pL2*z + \
        #         4*n*pL2*ps0*ps1*z + 2*n*p11*pL1*ps1*z - 8*n*p02*pL2*ps0*z - \
        #         4*n*p01*pL2*ps1*z - 2*n*p02*p11*pL1*z + 8*n*p01*p02*pL2*z + \
        #         n**2*p11*pL1*ps1*z - 2*p02*p11*pL1*z + p11*pL1*ps1*z - p11**3*z - \
        #         2*n**2*p01*pL2*ps0 - 4*n*p01*pL2*ps0 - n*p01*p11*pL1 - p01*p11*pL1 + \
        #         n*p11*pL1*ps0 + 2*n*pL2*ps0**2 + 2*n*p01**2*pL2 + p11*pL1*ps0 - p10*p11**2 + \
        #         n**2*pL2*ps0**2 + n**2*p01**2*pL2 + p11**2*pL0

        # # Convert the equation to a numerical function
        # polynomial_eq_func = lambda z_val: float(polynomial_eq.subs(z, z_val))
        # def find_root():
        #     # Bracketing interval for the root
        #     a, b = -1e15, 1e15  # Adjust the interval as needed
            
        #     # Define the numerical function
        #     def polynomial_eq_func(z_val):
        #         return float(polynomial_eq.subs(z, z_val))
            
        #     try:
        #         # Find the root using brentq method
        #         root = brentq(polynomial_eq_func, a, b)
        #         print(f"Root found at z = {root}")
        #     except ValueError:
        #         print("No root found in the interval.")
        
        # # Call the function to find the root
        # find_root()
        
    
    
    ########
    ## •	Select an appropriate solution as equilibrium prices
    ## •	Record and return equilibrium prices, equilibrium loading factors and equilibrium demand
        temp = np.array([price_l, price_s, x1, y1]).T
        
        temp = temp[~((temp[:, 0] < insurance_price_LB) | (temp[:, 1] < insurance_price_LB) | (temp[:, 2] < 0) | (temp[:, 3] < 0) |
                      (temp[:, 0] > insurance_price_UB) | (temp[:, 1] > insurance_price_UB))]
        
        if temp.size != 0:
            temp = temp[temp[:, 1] == max(temp[:, 1])]
            Eprice_perlvl_LR[n-1] = temp[0, 0]
            Eprice_perlvl_SFHA[n-1] = temp[0, 1]
            Edemand_perlvl_LR[n-1] = temp[0, 2]
            Edemand_perlvl_SFHA[n-1] = temp[0, 3]
            track_price.append([n, temp[0, 0], temp[0, 1], temp[0, 2]/1e8, temp[0, 3]/1e8])
    
    Elambda_perlvl_LR = Eprice_perlvl_LR - insurance_price_LB
    Elambda_perlvl_SFHA = Eprice_perlvl_SFHA - insurance_price_LB

    return Eprice_perlvl_LR, Eprice_perlvl_SFHA, Elambda_perlvl_LR, Elambda_perlvl_SFHA, Edemand_perlvl_LR, Edemand_perlvl_SFHA, track_price

