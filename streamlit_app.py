import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as mcolors
import time
import contextily as ctx
from shapely.geometry import Point, LineString
import folium
from streamlit_folium import st_folium
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the simulation function
def run_simulation(params, drone_positions, defender_positions, log_area):
    # Unpack parameters
    fov_angle = params['fov_angle']
    drone_speed = params['drone_speed']
    defender_range = params['defender_range']
    missile_speed = params['missile_speed']
    num_steps = params['num_steps']
    time_step = params['time_step']
    
    # Convert positions to numpy arrays
    drones = {
        'positions': np.array(drone_positions),
        'directions': np.random.rand(len(drone_positions)) * 360,
        'active': np.ones(len(drone_positions), dtype=bool),
        'paths': [[] for _ in range(len(drone_positions))]  # To store paths
    }
    defenders = {
        'positions': np.array(defender_positions),
        'ranges': np.full(len(defender_positions), defender_range / 111000),  # Convert meters to degrees
        'active': np.ones(len(defender_positions), dtype=bool)
    }
    missiles = []  # List to store active missiles
    
    # Initialize logs
    log_messages = []
    
    frames = []
    
    for step in range(num_steps):
        fig, ax = plt.subplots(figsize=(8, 8))
        # Set limits to the bounding box for Donetsk
        min_lon, max_lon = 37.5, 38.5
        min_lat, max_lat = 47.5, 48.5
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
    
        # Add base map
        ax.set_axis_off()
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    
        # Update drone positions and paths
        for i in range(len(drones['positions'])):
            if not drones['active'][i]:
                continue
            angle_rad = np.deg2rad(drones['directions'][i])
            dx = drone_speed * time_step * np.cos(angle_rad) / 111000  # Convert meters to degrees
            dy = drone_speed * time_step * np.sin(angle_rad) / 111000
            drones['positions'][i] += [dx, dy]
    
            # Ensure drones stay within map bounds
            drones['positions'][i][0] = np.clip(drones['positions'][i][0], min_lon, max_lon)
            drones['positions'][i][1] = np.clip(drones['positions'][i][1], min_lat, max_lat)
    
            # Append current position to path
            drones['paths'][i].append(drones['positions'][i].copy())
    
        # Update missile positions
        missiles_to_remove = []
        for missile in missiles:
            missile['position'] += missile['velocity']
            missile['path'].append(missile['position'].copy())
            # Check if missile has reached the target
            distance_to_target = np.linalg.norm(missile['position'] - missile['target'])
            if distance_to_target < (missile_speed * time_step) / 111000:
                # Missile has reached the target
                missiles_to_remove.append(missile)
                if missile['drone_index'] is not None and drones['active'][missile['drone_index']]:
                    drones['active'][missile['drone_index']] = False
                    log_msg = f"Step {step + 1}: Drone {missile['drone_index'] + 1} destroyed by Defender {missile['defender_index'] + 1}"
                    logger.info(log_msg)
                    log_messages.append(log_msg)
    
        # Remove missiles that have reached their targets
        for missile in missiles_to_remove:
            missiles.remove(missile)
    
        # Check for new missile launches
        for i in range(len(defenders['positions'])):
            if not defenders['active'][i]:
                continue
            defender_pos = defenders['positions'][i]
            defender_range = defenders['ranges'][i]
            for j in range(len(drones['positions'])):
                if not drones['active'][j]:
                    continue
                drone_pos = drones['positions'][j]
                distance = np.linalg.norm(drone_pos - defender_pos)
                if distance <= defender_range:
                    # Launch missile towards drone
                    missile_direction = (drone_pos - defender_pos) / distance
                    missile_velocity = missile_direction * (missile_speed * time_step) / 111000  # Convert meters to degrees
                    missile = {
                        'position': defender_pos.copy(),
                        'velocity': missile_velocity,
                        'target': drone_pos.copy(),
                        'path': [defender_pos.copy()],
                        'drone_index': j,
                        'defender_index': i
                    }
                    missiles.append(missile)
                    log_msg = f"Step {step + 1}: Defender {i + 1} launched missile at Drone {j + 1}"
                    logger.info(log_msg)
                    log_messages.append(log_msg)
                    # Defender can't launch another missile at the same drone in the same step
                    break  # Move to the next defender
    
        # Draw paths
        # Drone paths
        for i, path in enumerate(drones['paths']):
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], color='blue', alpha=0.5, linestyle='--')
    
        # Missile paths
        for missile in missiles:
            if len(missile['path']) > 1:
                path_array = np.array(missile['path'])
                ax.plot(path_array[:, 0], path_array[:, 1], color='orange', alpha=0.7)
    
        # Draw drones
        for i in range(len(drones['positions'])):
            if not drones['active'][i]:
                continue
            drone_pos = drones['positions'][i]
            wedge = Wedge(
                drone_pos,
                0.005,  # Adjust size as needed
                drones['directions'][i] - fov_angle / 2,
                drones['directions'][i] + fov_angle / 2,
                facecolor='blue',
                alpha=0.3,
                transform=ax.transData._b  # Ensure correct placement on map
            )
            ax.add_patch(wedge)
            ax.plot(*drone_pos, 'bo', markersize=5)
    
        # Draw defenders
        for i in range(len(defenders['positions'])):
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
    
        # Update log messages
        active_drones = np.sum(drones['active'])
        log_msg = f"Step {step + 1}: {active_drones} drones remaining."
        logger.info(log_msg)
        log_messages.append(log_msg)
    
        # Save the frame
        plt.title(f"Step {step + 1}")
        frames.append(fig)
        plt.close()
    
        # Update the log area
        log_area.text('\n'.join(log_messages))
    
        # Stop simulation if all drones are destroyed
        if active_drones == 0:
            log_msg = f"All drones have been destroyed at step {step + 1}."
            logger.info(log_msg)
            log_messages.append(log_msg)
            break  # Exit the simulation loop
    
    return frames
    
# Streamlit App
st.title("Air Defense Simulation over Donetsk")

# Tabs for parameters and simulation
tab1, tab2 = st.tabs(["Simulation Setup", "Simulation Results"])

with tab1:
    st.header("Set Simulation Parameters and Starting Positions")

    # Simulation parameters
    fov_angle = st.slider("Drone Field of View (degrees)", 30, 180, 120)
    drone_speed = st.slider("Drone Speed (meters per time step)", 100, 1000, 500)
    defender_range = st.slider("Defender Range (meters)", 1000, 10000, 5000)
    missile_speed = st.slider("Missile Speed (meters per time step)", 500, 2000, 1000)
    num_steps = st.slider("Number of Simulation Steps", 1, 100, 20)
    time_step = st.slider("Time Step Duration (seconds)", 1, 10, 5)

    params = {
        'fov_angle': fov_angle,
        'drone_speed': drone_speed,
        'defender_range': defender_range,
        'missile_speed': missile_speed,
        'num_steps': num_steps,
        'time_step': time_step
    }

    st.subheader("Set Starting Positions")
    st.write("Use the tools to add markers for Drones and Defenders on the map.")

    # Initialize map centered on Donetsk
    donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
    m = folium.Map(location=donetsk_coords, zoom_start=12)

    # Instructions for adding markers
    st.write("Select the type of marker to add, then click on the map to place it.")
    marker_type = st.radio("Select Marker Type to Add", ['Drone', 'Defender'])

    # Add existing markers to the map
    if 'drone_positions' not in st.session_state:
        st.session_state['drone_positions'] = []
    if 'defender_positions' not in st.session_state:
        st.session_state['defender_positions'] = []

    # Add existing drone markers
    for pos in st.session_state['drone_positions']:
        folium.Marker(
            location=[pos[1], pos[0]],
            icon=folium.Icon(color='blue', icon='plane', prefix='fa')
        ).add_to(m)

    # Add existing defender markers
    for pos in st.session_state['defender_positions']:
        folium.Marker(
            location=[pos[1], pos[0]],
            icon=folium.Icon(color='red', icon='tower', prefix='fa')
        ).add_to(m)

    # Capture map click events
    output = st_folium(m, width=700, height=500)

    # Handle map clicks
    if output and 'last_clicked' in output and output['last_clicked'] is not None:
        lat = output['last_clicked']['lat']
        lon = output['last_clicked']['lng']
        if marker_type == 'Drone':
            st.session_state['drone_positions'].append([lon, lat])
            st.success(f"Drone added at ({lat:.5f}, {lon:.5f})")
        elif marker_type == 'Defender':
            st.session_state['defender_positions'].append([lon, lat])
            st.success(f"Defender added at ({lat:.5f}, {lon:.5f})")
        # Clear last clicked to prevent duplicate entries
        output['last_clicked'] = None

    # Display current positions
    st.write(f"**Total Drones**: {len(st.session_state['drone_positions'])}")
    st.write(f"**Total Defenders**: {len(st.session_state['defender_positions'])}")

    # Start Simulation button
    if st.button("Start Simulation"):
        if len(st.session_state['drone_positions']) == 0 or len(st.session_state['defender_positions']) == 0:
            st.error("Please add at least one Drone and one Defender on the map.")
        else:
            st.session_state['params'] = params
            st.session_state['run_simulation'] = True
    else:
        st.session_state['run_simulation'] = False

    # Clear positions
    if st.button("Clear All Markers"):
        st.session_state['drone_positions'] = []
        st.session_state['defender_positions'] = []
        st.success("All markers have been cleared.")

with tab2:
    st.header("Simulation Results")

    if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
        # Create a placeholder for the logs
        log_area = st.empty()

        frames = run_simulation(
            st.session_state['params'],
            st.session_state['drone_positions'],
            st.session_state['defender_positions'],
            log_area
        )

        # Display frames
        for fig in frames:
            st.pyplot(fig)
            time.sleep(0.1)  # Control the speed of frame display

        # Clear positions after simulation
        st.session_state['drone_positions'] = []
        st.session_state['defender_positions'] = []
        st.session_state['run_simulation'] = False

    else:
        st.write("Please set the parameters and starting positions in the 'Simulation Setup' tab.")