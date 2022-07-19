#!usr/bin/env/python3

import numpy as np
from math import *

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

def pos_vel_from_spline(spline=[], s=0.0, boundaries=[0,1]):
    
    # Return x, y, vx, vy
    if s >= boundaries[0] and s <= boundaries[1]: return spline[0](s), spline[1](s), spline[2](s), spline[3](s)
    
    # Initial Position, 0 Velocities
    elif s < boundaries[0]: return spline[0](boundaries[0]), spline[1](boundaries[0]), 0.0, 0.0
    
    # Final Position, 0 Velocities
    elif s > boundaries[1]: return spline[0](boundaries[1]), spline[1](boundaries[1]), 0.0, 0.0

def check_position_from_goal(actual_pose=[0.0,0.0,0.0], b_target_pose=[1.0,1.0,pi], distance_threshold=0.2):
    
    # print(f'Distance From Target: {np.abs(np.array(actual_pose) - np.array(b_target_pose))}')
    if (np.abs(np.array(actual_pose) - np.array(b_target_pose)) < np.array([distance_threshold, distance_threshold, distance_threshold])).all(): return True
    else: return False
    
def plot_trajectory_graphs(xs, ys, xs_dot, ys_dot, s, T, S, DS, DX, DY):
    
    import matplotlib.pyplot as plt
    
    # Plot Trajectory in S
    plt.plot(xs(s), ys(s))
    plt.title('Original Path in S: X(s), Y(s)')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Plot Velocity in S
    plt.plot(s, xs_dot(s), s, ys_dot(s))
    plt.title('Speed in S: X_dot(s), Y_dot(s)')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Plot S in Time
    plt.plot(T, S)
    plt.title('S in Time')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Plot S_dot in Time
    plt.plot(T, DS)
    plt.title('S_dot in Time')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Plot Vx, Vy in Time
    plt.plot(T,np.multiply(DS,DX), T,np.multiply(DS,DY))
    plt.title('Vx, Vy in Time')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Plot Trajectory in Time
    plt.plot(T,xs(S), T,ys(S))
    plt.title("Trajectory in Time")
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Plot Trajectory with new S
    plt.plot(xs(S), ys(S))
    plt.title('New Path in S: X(S), Y(S)')
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Original Trajectory in S
    plt.plot(s,xs(s), s,ys(s))
    plt.title("Original Trajectory in S")
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    # Computed Trajectory in S
    plt.plot(S,xs(S), S,ys(S))
    plt.title("Computed Trajectory in S")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Computed Path
    plt.plot(xs(S), ys(S))
    plt.title("Computed Path")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def trajectory_in_t(path_in_s, b, θi, dt, ds=1, vel_max=[0.5,0.7]):

    xs, ys, xs_dot, ys_dot = path_in_s
    θ = θi
    
    # Get Linear and Angular Velocity Limits
    v_max, ω_max = vel_max
    
    s_n, t = 0.0, 0.0
    S, T, DS, DX, DY = [], [], [], [], []
    
    while s_n < 1:
        
        # Reset s_dot
        s_dot = ds
        
        # Append s, t
        S.append(s_n)
        T.append(t)
        
        # Compute Actual x_dot, y_dot
        dX, dY = xs_dot(s_n), ys_dot(s_n)
        
        # Compute Maximum Velocity
        vx_max = fabs(v_max * cos(θ) - b * ω_max * sin(θ))
        vy_max = fabs(v_max * sin(θ) + b * ω_max * cos(θ))
        
        # Compute s_dot Admissible
        s_dot_x = vx_max/abs(dX) if abs(dX*s_dot) > vx_max else s_dot
        s_dot_y = vy_max/abs(dY) if abs(dY*s_dot) > vy_max else s_dot
        s_dot = min(s_dot_x, s_dot_y, s_dot)
        
        # Save S_dot, X_dot, Y_dot
        DS.append(s_dot)
        DX.append(dX)
        DY.append(dY)

        # Compute New θ
        ω = 1/b * ((dY * s_dot) * cos(θ) - (dX * s_dot) * sin(θ))
        θ = ω * dt
        
        # Increase s, t
        s_n = s_n + s_dot*dt
        t = t + dt
    
    # All Plots
    # plot_trajectory_graphs(xs, ys, xs_dot, ys_dot, s, T, S, DS, DX, DY)
    
    # Return Vx, Vy splines and Final Time (t-dt)
    return IUS(T,xs(S)), IUS(T,ys(S)), IUS(T,np.multiply(DS,DX)), IUS(T,np.multiply(DS,DY)), t-dt

def check_velocity_limits(actual_velocities=[], max_vel=[0.5,0.7]):
    
    v, ω = actual_velocities
    v_max, ω_max = max_vel

    # Both Velocity Saturation    
    if fabs(v) > v_max and fabs(ω) > ω_max:
        print(f'Velocity Limits Exceeded | v = {v:.5f} | ω = {ω:.5f}')
        v, ω = np.sign(v) * v_max, np.sign(ω) * ω_max

    # Linear Velocity Saturation    
    elif fabs(v) > v_max:
        print(f'Linear Velocity Limit Exceeded | v = {v:.5f} | ω = {ω:.5f}')
        v = np.sign(v) * v_max

    # Angular Velocity Saturation    
    elif fabs(ω) > ω_max:
        print(f'Angular Velocity Limit Exceeded | v = {v:.5f} | ω = {ω:.5f}')
        ω = np.sign(ω) * ω_max

    return v, ω
