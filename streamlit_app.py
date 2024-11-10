import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import time
import contextily as ctx
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define real ranges for defenders
DEFENDER_TYPES = {
    'IRIS-T': {'range': 40000},    # in meters
    'Hawk SAM': {'range': 50000},
    'Patriot': {'range': 160000},
    'Gepard': {'range': 5500}
}

# Define attacker types
ATTACKER_TYPES = {
    'Drone': {'speed': 250},  # in meters per time step
    'Missile': {'speed': 1000},
    'Group of Drones': {'speed': 250}  # Group uses same speed as individual drone
}

# Define the simulation function
def run_simulation(params, attackers, defenders, target_position, log_area):
    # Unpack parameters
    time_step = params['time_step']
    num_steps = params['num_steps']
    avoidance_radius = params['avoidance_radius'] / 111000  # Convert meters to degrees

    # Initialize attackers
    attacker_positions = []
    attacker_speeds = []
    attacker_active = []
    attacker_paths = []
    attacker_types = []
    for attacker in attackers:
        attacker_positions.append(attacker['position'])
        attacker_speeds.append(ATTACKER_TYPES[attacker['type']]['speed'])
        attacker_active.append(True)
        attacker_paths.append([])
        attacker_types.append(attacker['type'])
    attacker_positions = np.array(attacker_positions)

    # Initialize defenders
    defender_positions = []
    defender_ranges = []
    defender_active = []
    defender_types = []
    for defender in defenders:
        defender_positions.append(defender['position'])
        defender_ranges.append(DEFENDER_TYPES[defender['type']]['range'] / 111000)  # Convert meters to degrees
        defender_active.append(True)
        defender_types.append(defender['type'])
    defender_positions = np.array(defender_positions)

    missiles = []  # List to store active missiles
    interception_points = []  # List to store interception points

    # Initialize logs
    log_messages = []

    frames = []
    current_time = 0  # Initialize simulation time

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

        # Plot the target
        if target_position:
            ax.plot(target_position[0], target_position[1], 'g*', markersize=15, label='Target')

        # Update attacker positions and paths
        for i in range(len(attacker_positions)):
            if not attacker_active[i]:
                continue

            current_pos = attacker_positions[i]

            # Calculate direction towards target
            if target_position:
                direction_vector = np.array(target_position) - current_pos
                distance_to_target = np.linalg.norm(direction_vector)
                if distance_to_target > 0:
                    direction_vector /= distance_to_target  # Normalize
                else:
                    direction_vector = np.array([0, 0])
            else:
                direction_vector = np.array([0, 0])  # No movement if no target

            # Avoidance logic
            for defender_pos, defender_range in zip(defender_positions, defender_ranges):
                distance_to_defender = np.linalg.norm(current_pos - defender_pos)
                if distance_to_defender < (defender_range + avoidance_radius):
                    # Calculate avoidance vector
                    avoidance_vector = current_pos - defender_pos
                    if np.linalg.norm(avoidance_vector) > 0:
                        avoidance_vector /= np.linalg.norm(avoidance_vector)
                        # Adjust the direction vector to avoid defender
                        direction_vector += avoidance_vector
                        direction_vector /= np.linalg.norm(direction_vector)

            # Move attacker
            speed = attacker_speeds[i]
            dx = speed * time_step * direction_vector[0] / 111000  # Convert meters to degrees
            dy = speed * time_step * direction_vector[1] / 111000
            attacker_positions[i] += [dx, dy]

            # Ensure attackers stay within map bounds
            attacker_positions[i][0] = np.clip(attacker_positions[i][0], min_lon, max_lon)
            attacker_positions[i][1] = np.clip(attacker_positions[i][1], min_lat, max_lat)

            # Append current position to path
            attacker_paths[i].append(attacker_positions[i].copy())

            # Check if attacker has reached the target
            if target_position and np.linalg.norm(attacker_positions[i] - target_position) < (speed * time_step) / 111000:
                attacker_active[i] = False
                log_msg = f"Time {current_time}s: {attacker_types[i]} {i + 1} has reached the target."
                logger.info(log_msg)
                log_messages.append(log_msg)

        # Update missile positions
        missiles_to_remove = []
        for missile in missiles:
            missile['position'] += missile['velocity']
            missile['path'].append(missile['position'].copy())
            # Check if missile has reached the target
            distance_to_target = np.linalg.norm(missile['position'] - missile['target'])
            if distance_to_target < (missile['speed'] * time_step) / 111000:
                # Missile has reached the target
                missiles_to_remove.append(missile)
                if missile['attacker_index'] is not None and attacker_active[missile['attacker_index']]:
                    attacker_active[missile['attacker_index']] = False
                    interception_points.append(attacker_positions[missile['attacker_index']].copy())
                    log_msg = f"Time {current_time}s: {defender_types[missile['defender_index']]} {missile['defender_index'] + 1} intercepted {attacker_types[missile['attacker_index']]} {missile['attacker_index'] + 1}"
                    logger.info(log_msg)
                    log_messages.append(log_msg)
            else:
                # Check if missile is out of range (simulate miss)
                missile_max_range = 200000 / 111000  # Missile max range in degrees (e.g., 200 km)
                missile_travel_distance = np.linalg.norm(missile['position'] - missile['start_position'])
                if missile_travel_distance >= missile_max_range:
                    missiles_to_remove.append(missile)
                    log_msg = f"Time {current_time}s: Missile from {defender_types[missile['defender_index']]} {missile['defender_index'] + 1} missed {attacker_types[missile['attacker_index']]} {missile['attacker_index'] + 1}"
                    logger.info(log_msg)
                    log_messages.append(log_msg)

        # Remove missiles that have reached their targets or max range
        for missile in missiles_to_remove:
            missiles.remove(missile)

        # Check for new missile launches
        for i in range(len(defender_positions)):
            if not defender_active[i]:
                continue
            defender_pos = defender_positions[i]
            defender_range = defender_ranges[i]
            for j in range(len(attacker_positions)):
                if not attacker_active[j]:
                    continue
                attacker_pos = attacker_positions[j]
                distance = np.linalg.norm(attacker_pos - defender_pos)
                if distance <= defender_range:
                    # Launch missile towards attacker
                    missile_direction = (attacker_pos - defender_pos) / distance
                    missile_speed = 2000  # Set missile speed (can be adjusted per defender type)
                    missile_velocity = missile_direction * missile_speed * time_step / 111000  # Convert meters to degrees
                    missile = {
                        'position': defender_pos.copy(),
                        'start_position': defender_pos.copy(),
                        'velocity': missile_velocity,
                        'target': attacker_pos.copy(),
                        'path': [defender_pos.copy()],
                        'attacker_index': j,
                        'defender_index': i,
                        'speed': missile_speed,
                    }
                    missiles.append(missile)
                    log_msg = f"Time {current_time}s: {defender_types[i]} {i + 1} launched missile at {attacker_types[j]} {j + 1}"
                    logger.info(log_msg)
                    log_messages.append(log_msg)
                    # Defender can't launch another missile at the same attacker in the same step
                    break  # Move to the next defender

        # Draw paths
        # Attacker paths
        for i, path in enumerate(attacker_paths):
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], color='blue', alpha=0.5, linestyle='--')

        # Missile paths
        for missile in missiles:
            if len(missile['path']) > 1:
                path_array = np.array(missile['path'])
                ax.plot(path_array[:, 0], path_array[:, 1], color='orange', alpha=0.7)

        # Draw interception points
        for point in interception_points:
            ax.plot(point[0], point[1], 'rx', markersize=10, label='Interception')

        # Draw attackers
        for i in range(len(attacker_positions)):
            if not attacker_active[i]:
                continue
            attacker_pos = attacker_positions[i]
            # Draw attacker as a blue dot
            ax.plot(*attacker_pos, 'bo', markersize=5)
            # Optionally, add labels or icons based on attacker type

        # Draw defenders
        for i in range(len(defender_positions)):
            if not defender_active[i]:
                continue
            defender_pos = defender_positions[i]
            circle = Circle(
                defender_pos,
                defender_ranges[i],
                color='red',
                alpha=0.1,
                transform=ax.transData._b
            )
            ax.add_patch(circle)
            ax.plot(*defender_pos, 'ro', markersize=5)
            # Optionally, add labels or icons based on defender type

        # Add timestamp to frame
        ax.text(0.01, 0.99, f'Time: {current_time}s', transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

        # Update log messages
        active_attackers = np.sum(attacker_active)
        log_msg = f"Time {current_time}s: {active_attackers} attackers remaining."
        logger.info(log_msg)
        log_messages.append(log_msg)

        # Save the frame
        plt.title(f"Simulation Time: {current_time}s")
        frames.append(fig)
        plt.close()

        # Update the log area
        log_area.text('\n'.join(log_messages[-10:]))  # Show last 10 messages

        # Increment simulation time
        current_time += time_step

        # Stop simulation if all attackers have reached the target or are destroyed
        if active_attackers == 0:
            log_msg = f"All attackers have reached the target or been destroyed at time {current_time}s."
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
    avoidance_radius = st.slider("Defender Avoidance Radius (meters)", 0, 5000, 1000)
    num_steps = st.slider("Number of Simulation Steps", 1, 100, 50)
    time_step = st.slider("Time Step Duration (seconds)", 1, 10, 5)

    params = {
        'avoidance_radius': avoidance_radius,
        'num_steps': num_steps,
        'time_step': time_step
    }

    st.subheader("Set Starting Positions and Target")
    st.write("Use the tools to add markers for Attackers, Defenders, and the Target on the map.")

    # Initialize map centered on Donetsk
    donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
    m = folium.Map(location=donetsk_coords, zoom_start=12)

    # Instructions for adding markers
    marker_type = st.radio("Select Marker Type to Add", ['Attacker', 'Defender', 'Target'])

    # Selection of attacker or defender type
    if marker_type == 'Attacker':
        attacker_type = st.selectbox("Select Attacker Type", list(ATTACKER_TYPES.keys()))
        if attacker_type == 'Group of Drones':
            st.write("Click on the map to place multiple drones as a group.")
    elif marker_type == 'Defender':
        defender_type = st.selectbox("Select Defender Type", list(DEFENDER_TYPES.keys()))

    # Add existing markers to the map
    if 'attackers' not in st.session_state:
        st.session_state['attackers'] = []
    if 'defenders' not in st.session_state:
        st.session_state['defenders'] = []
    if 'target_position' not in st.session_state:
        st.session_state['target_position'] = None

    # Add existing attacker markers
    for attacker in st.session_state['attackers']:
        icon = 'plane' if attacker['type'] != 'Group of Drones' else 'fighter-jet'
        folium.Marker(
            location=[attacker['position'][1], attacker['position'][0]],
            icon=folium.Icon(color='blue', icon=icon, prefix='fa'),
            popup=f"Attacker: {attacker['type']}"
        ).add_to(m)

    # Add existing defender markers
    for defender in st.session_state['defenders']:
        folium.Marker(
            location=[defender['position'][1], defender['position'][0]],
            icon=folium.Icon(color='red', icon='shield', prefix='fa'),
            popup=f"Defender: {defender['type']}"
        ).add_to(m)

    # Add target marker
    if st.session_state['target_position']:
        folium.Marker(
            location=[st.session_state['target_position'][1], st.session_state['target_position'][0]],
            icon=folium.Icon(color='green', icon='star', prefix='fa')
        ).add_to(m)

    # Capture map click events
    output = st_folium(m, width=700, height=500)

    # Handle map clicks
    if output and 'last_clicked' in output and output['last_clicked'] is not None:
        lat = output['last_clicked']['lat']
        lon = output['last_clicked']['lng']
        if marker_type == 'Attacker':
            if 'attacker_type' not in st.session_state:
                st.session_state['attacker_type'] = attacker_type
            if attacker_type == 'Group of Drones':
                # Allow multiple clicks to add drones to the group
                st.session_state['attackers'].append({
                    'type': 'Drone',  # Each drone is added as an individual attacker
                    'position': [lon, lat]
                })
                st.success(f"Drone added to group at ({lat:.5f}, {lon:.5f})")
            else:
                st.session_state['attackers'].append({
                    'type': attacker_type,
                    'position': [lon, lat]
                })
                st.success(f"{attacker_type} attacker added at ({lat:.5f}, {lon:.5f})")
        elif marker_type == 'Defender':
            st.session_state['defenders'].append({
                'type': defender_type,
                'position': [lon, lat]
            })
            st.success(f"{defender_type} defender added at ({lat:.5f}, {lon:.5f})")
        elif marker_type == 'Target':
            st.session_state['target_position'] = [lon, lat]
            st.success(f"Target set at ({lat:.5f}, {lon:.5f})")
        # Clear last clicked to prevent duplicate entries
        output['last_clicked'] = None

    # Display current positions
    st.write(f"**Total Attackers**: {len(st.session_state['attackers'])}")
    st.write(f"**Total Defenders**: {len(st.session_state['defenders'])}")
    st.write(f"**Target Set**: {'Yes' if st.session_state['target_position'] else 'No'}")

    # Start Simulation button
    if st.button("Start Simulation"):
        if len(st.session_state['attackers']) == 0 or len(st.session_state['defenders']) == 0 or not st.session_state['target_position']:
            st.error("Please add at least one Attacker, one Defender, and set the Target on the map.")
        else:
            st.session_state['params'] = params
            st.session_state['run_simulation'] = True
    else:
        st.session_state['run_simulation'] = False

    # Clear positions
    if st.button("Clear All Markers"):
        st.session_state['attackers'] = []
        st.session_state['defenders'] = []
        st.session_state['target_position'] = None
        st.success("All markers have been cleared.")

with tab2:
    st.header("Simulation Results")

    if 'run_simulation' in st.session_state and st.session_state['run_simulation']:
        # Create a placeholder for the logs
        log_area = st.empty()

        frames = run_simulation(
            st.session_state['params'],
            st.session_state['attackers'],
            st.session_state['defenders'],
            st.session_state['target_position'],
            log_area
        )

        # Display frames
        for fig in frames:
            st.pyplot(fig)
            time.sleep(0.1)  # Control the speed of frame display

        # Clear positions after simulation
        st.session_state['attackers'] = []
        st.session_state['defenders'] = []
        st.session_state['target_position'] = None
        st.session_state['run_simulation'] = False

    else:
        st.write("Please set the parameters and starting positions in the 'Simulation Setup' tab.")