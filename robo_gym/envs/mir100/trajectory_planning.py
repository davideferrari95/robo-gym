#!usr/bin/env/python3

import numpy as np
from math import *
from robo_gym.envs.mir100.utils import angular_difference_radians

# Import 1D Spline Library
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

''' Path Planning Equations | s € [0,1] - Differential Flatness
    
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

def plot_trajectory(x, y, s=np.linspace(0, 1, 101), seconds=3):

    import matplotlib.pyplot as plt
    
    # Plot Trajectory
    plt.plot(x(s), y(s))
    
    # Set Plot Limits
    x_min, y_min = min([x(i) for i in s]), min([y(i) for i in s])
    x_max, y_max = max([x(i) for i in s]), max([y(i) for i in s])
    plt.xlim([x_min-1, x_max+1])
    plt.ylim([y_min-1, y_max+1])
    
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
            s = np.linspace(0, 1, 10001)
            x = -(s-1)**3 * xi + s**3 * xf + αx * s**2 * (s-1) + βx * s * (s-1)**2
            y = -(s-1)**3 * yi + s**3 * yf + αy * s**2 * (s-1) + βy * s * (s-1)**2
            x_s, y_s = IUS(s, x), IUS(s, y)
            vx_s, vy_s = x_s.derivative(), y_s.derivative()

            # Plot Trajectory
            plot_trajectory(x_s, y_s, s, seconds=3)
            
            return x_s, y_s, vx_s, vy_s
        
        elif parametrization in ['t']:
            
            # Spline Interpolation in t € [0,tf]
            t = np.linspace(0, tf, 10001)
            x = -((t-tf)/tf)**3 * xi + (t/tf)**3 * xf + αx * (t/tf)**2 * ((t-tf)/tf) + βx * t/tf * ((t-tf)/tf)**2
            y = -((t-tf)/tf)**3 * yi + (t/tf)**3 * yf + αy * (t/tf)**2 * ((t-tf)/tf) + βy * t/tf * ((t-tf)/tf)**2
            x_t, y_t = IUS(t, x), IUS(t, y)
            vx_t, vy_t = x_t.derivative(), y_t.derivative()

            # Plot Trajectory
            plot_trajectory(x_t, y_t, t, seconds=3)
            
            return x_t, y_t, vx_t, vy_t

def pos_vel_from_spline(spline, s, boundaries=[0,1]):
    
    # Return x, y, vx, vy
    if s >= boundaries[0] and s <= boundaries[1]: return spline[0](s), spline[1](s), spline[2](s), spline[3](s)
    
    # Initial Position, 0 Velocities
    elif s < boundaries[0]: return spline[0](boundaries[0]), spline[1](boundaries[0]), 0.0, 0.0
    
    # Final Position, 0 Velocities
    elif s > boundaries[1]: return spline[0](boundaries[1]), spline[1](boundaries[1]), 0.0, 0.0
    
def velocity_saturation(actual_state, desired_state, dt, b=0.2, max_vel=[0.5,0.7]):
    
    x, y, θ, v, ω = actual_state
    x_des, y_des, Vbx_des, Vby_des = desired_state
    
    # Assume Small ω Changes on Path (ω_des ~= ω) 
    θ_des = θ + ω*dt
    Δθ = angular_difference_radians(θ_des, θ)
    # print(f'θ: {θ} | θ_des: {θ_des} | Δθ: {Δθ}\nθ: {degrees(θ)} | θ_des: {degrees(θ_des)} | Δθ: {degrees(Δθ)}')
    
    vx_max, vy_max = max_vel[0] * cos(Δθ) - max_vel[1] * b*sin(Δθ), max_vel[0] * sin(Δθ) + max_vel[1] * b*cos(Δθ)
    scale_factor = max(fabs(Vbx_des/vx_max), fabs(Vby_des/vy_max))

    return np.multiply([Vbx_des,Vby_des], (1 / scale_factor))

def check_position_from_goal(actual_pose=[0.0,0.0,0.0], b_target_pose=[1.0,1.0,pi], distance_threshold=0.2):
    
    # print(f'Distance From Target: {np.abs(np.array(actual_pose) - np.array(b_target_pose))}')
    if (np.abs(np.array(actual_pose) - np.array(b_target_pose)) < np.array([distance_threshold, distance_threshold, distance_threshold])).all(): return True
    else: return False
    
# def compute_ds(state, spline, s, b, max_vel):
    
#     θ, v, ω = state
#     v_max, ω_max = max_vel
#     x, y, vx, vy = spline
    
#     return min((v_max*cos(θ)-b*ω_max*sin(θ))/vx(s), (v_max*sin(θ)+b*ω_max*cos(θ))/vy(s))

# def compute_ds(dt, spline, max_vel):
    
#     # Get Maximum Velocity Parameters
#     v_max, ω_max = max_vel
    
#     # Get Splines
#     x, y, vx, vy = spline
#     trajectory_lenght = 0.0
#     L = 10000
    
#     # Compute Trajectory Lenght
#     for s in range(1, L): trajectory_lenght += sqrt(pow(x(s/L)-x((s-1)/L),2) + pow(y(s/L)-y((s-1)/L),2))
    
#     distance_in_dt = v_max * dt
#     samples = trajectory_lenght / distance_in_dt
#     ds = (1 / samples)
    
#     print(f'Trajectory Lenght: {trajectory_lenght:.4f} | ds: {ds:.4f}')
    
#     return ds