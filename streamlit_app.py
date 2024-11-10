import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from streamlit import caching
import time
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

# Define the simulation function
def run_simulation(params):
    # Unpack parameters
    num_drones = params['num_drones']
    num_defenders = params['num_defenders']
    fov_angle = params['fov_angle']
    drone_speed = params['drone_speed']
    defender_range = params['defender_range']
    num_steps = params['num_steps']
    time_step = params['time_step']

    # Define the bounding box for Donetsk (approximate coordinates)
    min_lon, max_lon = 37.5, 38.5
    min_lat, max_lat = 47.5, 48.5

    # Initialize positions
    drones = {
        'positions': np.column_stack((
            np.random.uniform(min_lon, max_lon, num_drones),
            np.random.uniform(min_lat, max_lat, num_drones)
        )),
        'directions': np.random.rand(num_drones) * 360,
        'active': np.ones(num_drones, dtype=bool)
    }
    defenders = {
        'positions': np.column_stack((
            np.random.uniform(min_lon, max_lon, num_defenders),
            np.random.uniform(min_lat, max_lat, num_defenders)
        )),
        'ranges': np.full(num_defenders, defender_range / 111000),  # Convert meters to degrees
        'active': np.ones(num_defenders, dtype=bool)
    }

    frames = []

    for step in range(num_steps):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Set limits to the bounding box
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

        # Add base map
        ax.set_axis_off()
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)

        # Update drone positions
        for i in range(num_drones):
            if not drones['active'][i]:
                continue
            angle_rad = np.deg2rad(drones['directions'][i])
            dx = drone_speed * time_step * np.cos(angle_rad) / 111000  # Convert meters to degrees
            dy = drone_speed * time_step * np.sin(angle_rad) / 111000
            drones['positions'][i] += [dx, dy]

            # Wrap around map edges
            drones['positions'][i][0] = np.clip(drones['positions'][i][0], min_lon, max_lon)
            drones['positions'][i][1] = np.clip(drones['positions'][i][1], min_lat, max_lat)

            # Draw drones
            drone_pos = drones['positions'][i]
            wedge = Wedge(
                drone_pos,
                0.001,  # Adjust size as needed
                drones['directions'][i] - fov_angle / 2,
                drones['directions'][i] + fov_angle / 2,
                facecolor='blue',
                alpha=0.3,
                transform=ax.transData._b  # Ensure correct placement on map
            )
            ax.add_patch(wedge)
            ax.plot(*drone_pos, 'bo', markersize=5)

        # Draw defenders
        for i in range(num_defenders):
            if not defenders['active'][i]:
                continue
            defender_pos = defenders['positions'][i]
            circle = Circle(
                defender_pos,
                defenders['ranges'][i],
                color='red',
                alpha=0.1,
                transform=ax.transData._b
            )
            ax.add_patch(circle)
            ax.plot(*defender_pos, 'ro', markersize=5)

        # Check for interactions
        for i in range(num_drones):
            if not drones['active'][i]:
                continue
            drone_point = Point(drones['positions'][i])
            for j in range(num_defenders):
                if not defenders['active'][j]:
                    continue
                defender_point = Point(defenders['positions'][j])
                distance = drone_point.distance(defender_point) * 111000  # Convert degrees to meters
                if distance <= defender_range:
                    # Drone is destroyed
                    drones['active'][i] = False
                    break

        # Save the frame
        plt.title(f"Step {step + 1}")
        frames.append(fig)
        plt.close()

    return frames

# Streamlit App
st.title("Air Defense Simulation over Donetsk")

# Tabs for parameters and simulation
tab1, tab2 = st.tabs(["Simulation Parameters", "Simulation Results"])

with tab1:
    st.header("Set Simulation Parameters")

    # Simulation parameters
    num_drones = st.slider("Number of Drones", 1, 20, 5)
    num_defenders = st.slider("Number of Air Defense Units", 1, 20, 5)
    fov_angle = st.slider("Drone Field of View (degrees)", 30, 180, 120)
    drone_speed = st.slider("Drone Speed (meters per time step)", 100, 1000, 500)
    defender_range = st.slider("Defender Range (meters)", 1000, 10000, 5000)
    num_steps = st.slider("Number of Simulation Steps", 1, 100, 50)
    time_step = st.slider("Time Step Duration (seconds)", 1, 10, 5)

    params = {
        'num_drones': num_drones,
        'num_defenders': num_defenders,
        'fov_angle': fov_angle,
        'drone_speed': drone_speed,
        'defender_range': defender_range,
        'num_steps': num_steps,
        'time_step': time_step
    }

    if st.button("Start Simulation"):
        # Clear cache to reset simulation
        caching.clear_cache()
        st.session_state['params'] = params
        st.session_state['run_simulation'] = True
    else:
        st.session_state['run_simulation'] = False

with tab2:
    st.header("Simulation Results")

    if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
        frames = run_simulation(st.session_state['params'])
        for fig in frames:
            st.pyplot(fig)
            time.sleep(0.1)  # Control the speed of frame display
    else:
        st.write("Please set the parameters and start the simulation in the 'Simulation Parameters' tab.")