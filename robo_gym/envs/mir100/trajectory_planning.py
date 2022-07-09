#!usr/bin/env/python3

import numpy as np
from math import *

# Import 1D Spline Library
from scipy.interpolate import InterpolatedUnivariateSpline

''' Path Planning Equations | s € [0,1]
    
        xs = -pow(s-1,3) * xi + pow(s,3) * xf + αx * pow(s,2) * (s-1) + βx * s * pow(s-1,2)
        ys = -pow(s-1,3) * yi + pow(s,3) * yf + αy * pow(s,2) * (s-1) + βy * s * pow(s-1,2)
    
    Boundary Conditions
    
        x(0) = xi | y(0) = yi
        x(1) = xf | y(1) = yf
    
    Orientation Conditions
    
        x'(0) = Ki * cos(θi) | y'(0) = Ki * sin(θi)
        x'(1) = Kf * cos(θf) | y'(1) = Kf * sin(θf)
        Ki = Kf = K > 0
        
    Orientation Equations
    
        αx = K * cos(θf) - 3xf | αy = K * sin(θf) + 3yf
        βx = K * cos(θi) - 3xi | βy = K * sin(θi) + 3yi
'''      
''' Trajectory Tracking - PD + Feedforward
    
        u1 = xd'' + Kp1 * (xd - x) + Kd1 * (xd' - x')
        u2 = yd'' + Kp2 * (yd - y) + Kd2 * (yd' - y')
        v' = u1 * cos(θ) + u2 * sin(θ)
        ω = (u2 * cos(θ) - u1 * sin(θ)) / v
        Kp1, Kp2, Kd1, Kd2 > 0
'''
''' Trajectory Tracking - IO-SFL
    
        xb, yb = (x + b * cos(θ)), (y + b * sin(θ))
        ex, ey = (x_des - xb), (y_des - yb)
        Vbx, Vby = (Vx_des + k1 * ex), (Vy_des + k2 * ey)
        v = Vbx * cos(θ) + Vby * sin(θ)
        ω = 1/b * (Vby * cos(θ) - Vbx * sin(θ))
'''

def plot_trajectory(x, y, i, seconds=3):
            
    import matplotlib.pyplot as plt
    
    # Plot Trajectory
    plt.plot(x(i), y(i))
    plt.show(block=False)
    plt.pause(seconds)
    plt.close()

def plan_trajectory(parametrization = 's', method = '3rd polynomial', start=[0.0,0.0,0.0], target=[1.0,1.0,pi], b=0.2, parameters=[]):
    
    
    # Get Trajectory Parameters
    xi, yi, θi, xf, yf, θf = start + target

    # 3rd Order Polynomial Path
    if method == '3rd polynomial':
        
        # Check Input Parameters
        if   parametrization in ['s'] and len(parameters) == 1: K = parameters[0]
        elif parametrization in ['t'] and len(parameters) == 2: K, tf = parameters
        else: print('ERROR: Trajectory Planning | Invalid Parameter Number'); exit()
            
        # Compute Trajectory Parameters
        αx = K * cos(θf) - 3 * xf
        αy = K * sin(θf) - 3 * yf
        βx = K * cos(θi) + 3 * xi
        βy = K * sin(θi) + 3 * yi
        
        if parametrization in ['s']: 
            
            # Spline Interpolation in s € [0,1]
            s = np.linspace(0, 1, 10000)
            x = -(s-1)**3 * xi + s**3 * xf + αx * s**2 * (s-1) + βx * s * (s-1)**2
            y = -(s-1)**3 * yi + s**3 * yf + αy * s**2 * (s-1) + βy * s * (s-1)**2
            x_s, y_s = InterpolatedUnivariateSpline(s, x), InterpolatedUnivariateSpline(s, y)
            vx_s, vy_s = x_s.derivative(), y_s.derivative()
            
            # Plot Trajectory
            plot_trajectory(x_s, y_s, s, 3)
            
            return x_s, y_s, vx_s, vy_s
        
        elif parametrization in ['t']:
            
            # Spline Interpolation in t € [0,tf]
            t = np.linspace(0, tf, 10000)
            x = -((t-tf)/tf)**3 * xi + (t/tf)**3 * xf + αx * (t/tf)**2 * ((t-tf)/tf) + βx * t/tf * ((t-tf)/tf)**2
            y = -((t-tf)/tf)**3 * yi + (t/tf)**3 * yf + αy * (t/tf)**2 * ((t-tf)/tf) + βy * t/tf * ((t-tf)/tf)**2
            x_t, y_t = InterpolatedUnivariateSpline(t, x), InterpolatedUnivariateSpline(t, y)
            vx_t, vy_t = x_t.derivative(), y_t.derivative()

            # Plot Trajectory
            plot_trajectory(x_t, y_t, t, 3)
            
            return x_t, y_t, vx_t, vy_t

def pos_vel_from_spline(spline, i):
    
    x, y   = spline[0](i), spline[1](i)
    vx, vy = spline[2](i), spline[3](i)
    
    return x, y, vx, vy

def compute_ds(dt, xi, yi, xf, yf, max_vel=[]):
    
    trajectory_lenght = sqrt(pow(xf-xi,2) + pow(yf-yi,2))
    execution_vel = max_vel[0]
    distance_in_dt = execution_vel * dt
    samples = trajectory_lenght / distance_in_dt
    ds = (1 / samples)
    
    print(f'Trajectory Lenght: {trajectory_lenght:.4f} | ds: {ds:.4f}')
    
    return ds

def compute_maximum_velocity(θ, b, max_vel):
    
    v_max, ω_max = max_vel
    
    vx_max = v_max * cos(θ) - ω_max * b * sin(θ)
    vy_max = v_max * sin(θ) - ω_max * b * cos(θ)
    
    return fabs(vx_max), fabs(vy_max)

def velocity_saturation(v, vmax):
    
    vx, vy = v
    vx_max, vy_max = vmax
    
    scale_factor = max(fabs(vx/vx_max), fabs(vy/vy_max))
    
    return np.multiply([vx, vy], (1 / scale_factor))

def check_position_from_goal(actual_pose=[0.0,0.0,0.0], b_target_pose=[1.0,1.0,pi], distance_threshold=0.2):
    
    if (np.abs(np.array(actual_pose) - np.array(b_target_pose)) < np.array([distance_threshold, distance_threshold, distance_threshold/4])).all(): return True
    else: return False


''' def s_cubic_polynomial_path(self, s_, xi, yi, xf, yf, αx, αy, βx, βy):

        from scipy import misc

        # Compute Cubic Polynomial Path | s € [0,1]
        def x(s): return -pow(s-1,3) * xi + pow(s,3) * xf + αx * pow(s,2) * (s-1) + βx * s * pow(s-1,2)
        def y(s): return -pow(s-1,3) * yi + pow(s,3) * yf + αy * pow(s,2) * (s-1) + βy * s * pow(s-1,2)

        return x(s_), y(s_), misc.derivative(x,s_), misc.derivative(y,s_)

    def t_cubic_polynomial_path(self, t_, tf, xi, yi, xf, yf, αx, αy, βx, βy):

        from scipy import misc

        # Compute Cubic Polynomial Path | t € [0,tf]
        def x(t): return -pow((t-tf)/tf,3) * xi + pow(t/tf,3) * xf + αx * pow(t/tf,2) * ((t-tf)/tf) + βx * t/tf * pow((t-tf)/tf,2)
        def y(t): return -pow((t-tf)/tf,3) * yi + pow(t/tf,3) * yf + αy * pow(t/tf,2) * ((t-tf)/tf) + βy * t/tf * pow((t-tf)/tf,2)

        return x(t_), y(t_), misc.derivative(x,t_), misc.derivative(y,t_)
'''        

''' def s_plot_trajectory(self, xi, yi, xf, yf, αx, αy, βx, βy):
    
        import matplotlib.pyplot as plt

        x, y = [], []
        samples = 10000 #100000
        for i in range(0, samples):
            s = i/samples
            x_, y_, vx, vy = self.s_cubic_polynomial_path(s, xi, yi, xf, yf, αx, αy, βx, βy)
            x.append(x_)
            y.append(y_)

        plt.scatter(x,y)
        plt.xlim([min(xi,yi,xf,yf)-1, max(xi,yi,xf,yf)+1])
        plt.ylim([min(xi,yi,xf,yf)-1, max(xi,yi,xf,yf)+1])
        plt.show(block=False) # plt.show()
        plt.pause(3)
        plt.close()

    def t_plot_trajectory(self, tf, xi, yi, xf, yf, αx, αy, βx, βy):

        import matplotlib.pyplot as plt

        x, y = [], []
        samples = 10000 #100000
        for i in range(0, samples):
            t = tf * i/samples
            x_, y_, vx, vy = self.t_cubic_polynomial_path(t, tf, xi, yi, xf, yf, αx, αy, βx, βy)
            x.append(x_)
            y.append(y_)

        plt.scatter(x,y)
        plt.xlim([min(xi,yi,xf,yf)-1, max(xi,yi,xf,yf)+1])
        plt.ylim([min(xi,yi,xf,yf)-1, max(xi,yi,xf,yf)+1])
        plt.show(block=False) # plt.show()
        plt.pause(3)
        plt.close()
'''        
