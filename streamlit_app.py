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

# Define mathematical functions for Defenders
def discovery_function(radar_cross_section, altitude, speed, distance):
    # Example mathematical model for discovery
    # Higher radar cross-section, lower altitude, lower speed, and shorter distance increase discovery probability
    return min(1, (radar_cross_section * (5000 - altitude) * (1000 - speed)) / (distance * 1e6))

def chance_to_kill(radar_cross_section, altitude, speed, distance):
    # Example mathematical model for kill chance
    # Higher radar cross-section, lower altitude, higher speed, and shorter distance increase kill chance
    return min(1, (radar_cross_section * (10000 - altitude) * speed) / (distance * 1e6))

# Define guidance functions for Attackers
def guidance_function(guidance_type, current_pos, target_pos):
    if guidance_type == 'GPS':
        # Direct line towards target
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.array([0, 0])
        return direction / distance
    elif guidance_type == 'Inertial':
        # Maintain current direction (simple inertial guidance)
        direction = current_pos - target_pos  # Placeholder for inertial logic
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.array([0, 0])
        return direction / distance
    elif guidance_type == 'Terrain Following':
        # Adjust path based on terrain (simplified as random adjustment)
        adjustment = np.random.uniform(-0.1, 0.1, size=2)
        direction = (target_pos - current_pos) / np.linalg.norm(target_pos - current_pos) + adjustment
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.array([0, 0])
        return direction / distance
    else:
        return np.array([0, 0])

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
    attacker_guidances = []
    for attacker in attackers:
        attacker_positions.append(attacker['position'])
        attacker_speeds.append(ATTACKER_TYPES[attacker['type']]['speed'])
        attacker_active.append(True)
        attacker_paths.append([])
        attacker_types.append(attacker['type'])
        attacker_guidances.append(attacker.get('guidance', 'GPS'))
    attacker_positions = np.array(attacker_positions)

    # Initialize defenders
    defender_positions = []
    defender_ranges = []
    defender_active = []
    defender_types = []
    defender_missile_number = []
    defender_max_targets = []
    defender_missile_count = []
    for defender in defenders:
        defender_positions.append(defender['position'])
        defender_ranges.append(DEFENDER_TYPES[defender['type']]['range'] / 111000)  # Convert meters to degrees
        defender_active.append(True)
        defender_types.append(defender['type'])
        defender_missile_number.append(defender.get('missile_number', 8))
        defender_max_targets.append(defender.get('max_targets', 2))
        defender_missile_count.append(0)
    defender_positions = np.array(defender_positions)

    missiles = []  # List to store active missiles
    interception_points = []  # List to store interception points

    # Initialize logs
    log_messages = []

    frames = []
    current_time = 0  # Initialize simulation time

    for step in range(num_steps):
        fig, ax = plt.subplots(figsize=(10, 10))
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
            ax.plot(target_position[0], target_position[1], 'g*', markersize=20, label='Target')

        # Update attacker positions and paths
        for i in range(len(attacker_positions)):
            if not attacker_active[i]:
                continue

            current_pos = attacker_positions[i]

            # Calculate direction towards target with guidance
            if target_position:
                direction_vector = guidance_function(attacker_guidances[i], current_pos, target_position)
                distance_to_target = np.linalg.norm(target_position - current_pos)
                if distance_to_target > 0:
                    direction_vector = direction_vector / np.linalg.norm(direction_vector)
                else:
                    direction_vector = np.array([0, 0])
            else:
                direction_vector = np.array([0, 0])  # No movement if no target

            # Avoidance logic
            for j, (defender_pos, defender_range) in enumerate(zip(defender_positions, defender_ranges)):
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
                    # Apply chance to kill
                    attacker = attackers[missile['attacker_index']]
                    defender = defenders[missile['defender_index']]
                    discovery_prob = discovery_function(
                        radar_cross_section=attacker.get('radar_cross_section', 1),
                        altitude=attacker.get('altitude', 1000),
                        speed=attacker_speeds[missile['attacker_index']],
                        distance=np.linalg.norm(attacker_positions[missile['attacker_index']] - defender_positions[missile['defender_index']])
                    )
                    kill_prob = chance_to_kill(
                        radar_cross_section=attacker.get('radar_cross_section', 1),
                        altitude=attacker.get('altitude', 1000),
                        speed=attacker_speeds[missile['attacker_index']],
                        distance=np.linalg.norm(attacker_positions[missile['attacker_index']] - defender_positions[missile['defender_index']])
                    )
                    if np.random.rand() < discovery_prob and np.random.rand() < kill_prob:
                        attacker_active[missile['attacker_index']] = False
                        interception_points.append(missile['position'].copy())
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
            targets_tracked = 0
            for j in range(len(attacker_positions)):
                if not attacker_active[j]:
                    continue
                attacker_pos = attacker_positions[j]
                distance = np.linalg.norm(attacker_pos - defender_pos)
                if distance <= defender_range and defender_missile_count[i] < defender_missile_number[i]:
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
                    defender_missile_count[i] += 1
                    log_msg = f"Time {current_time}s: {defender_types[i]} {i + 1} launched missile at {attacker_types[j]} {j + 1}"
                    logger.info(log_msg)
                    log_messages.append(log_msg)
                    targets_tracked += 1
                    if targets_tracked >= defender_max_targets[i]:
                        break  # Defender reached max tracking for this step

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
            ax.plot(point[0], point[1], 'rx', markersize=12, label='Interception')

        # Draw attackers
        for i in range(len(attacker_positions)):
            if not attacker_active[i]:
                continue
            attacker_pos = attacker_positions[i]
            # Draw attacker as a blue dot
            ax.plot(*attacker_pos, 'bo', markersize=8)
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
            ax.plot(*defender_pos, 'ro', markersize=8)
            # Optionally, add labels or icons based on defender type

        # Add timestamp to frame
        ax.text(0.01, 0.99, f'Time: {current_time}s', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Update log messages
        active_attackers = np.sum(attacker_active)
        log_msg = f"Time {current_time}s: {active_attackers} attackers remaining."
        logger.info(log_msg)
        log_messages.append(log_msg)

        # Save the frame
        plt.title(f"Simulation Time: {current_time}s")
        frames.append(fig)
        plt.close()

        # Update the log area (show last 10 messages)
        log_area.text('\n'.join(log_messages[-10:]))

        # Increment simulation time
        current_time += time_step

        # Reset missile counts for defenders for the next step
        for i in range(len(defender_missile_count)):
            defender_missile_count[i] = 0

        # Stop simulation if all attackers have reached the target or are destroyed
        if active_attackers == 0:
            log_msg = f"All attackers have reached the target or been destroyed at time {current_time}s."
            logger.info(log_msg)
            log_messages.append(log_msg)
            break  # Exit the simulation loop

    return frames

# Streamlit App
st.title("Advanced Air Defense Simulation over Donetsk")

# Initialize session state for attackers and defenders
if 'attackers' not in st.session_state:
    st.session_state['attackers'] = []
if 'defenders' not in st.session_state:
    st.session_state['defenders'] = []
if 'target_position' not in st.session_state:
    st.session_state['target_position'] = None

# Tabs for Simulation Setup and Results
tab1, tab2 = st.tabs(["Simulation Setup", "Simulation Results"])

with tab1:
    st.header("Configure Simulation Parameters and Entities")

    # Simulation parameters
    st.subheader("General Parameters")
    avoidance_radius = st.slider("Defender Avoidance Radius (meters)", 0, 5000, 1000)
    num_steps = st.slider("Number of Simulation Steps", 1, 100, 50)
    time_step = st.slider("Time Step Duration (seconds)", 1, 10, 5)

    params = {
        'avoidance_radius': avoidance_radius,
        'num_steps': num_steps,
        'time_step': time_step
    }

    st.markdown("---")

    # Tabs within Simulation Setup for Attackers and Defenders
    setup_tab1, setup_tab2, setup_tab3 = st.tabs(["Add Attackers", "Add Defenders", "Set Target"])

    with setup_tab1:
        st.subheader("Add Attackers")
        attacker_type = st.selectbox("Select Attacker Type", list(ATTACKER_TYPES.keys()))
        if attacker_type == 'Group of Drones':
            st.info("Select multiple points on the map to add a group of drones.")
        else:
            st.info("Select a point on the map to add an attacker.")

        # Initialize map centered on Donetsk
        donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
        m = folium.Map(location=donetsk_coords, zoom_start=12)

        # Add existing attacker markers
        for attacker in st.session_state['attackers']:
            if attacker['type'] == 'Drone':
                icon = 'plane'
            elif attacker['type'] == 'Missile':
                icon = 'rocket'
            else:
                icon = 'fighter-jet'
            folium.Marker(
                location=[attacker['position'][1], attacker['position'][0]],
                icon=folium.Icon(color='blue', icon=icon, prefix='fa'),
                popup=f"Attacker: {attacker['type']}"
            ).add_to(m)

        # Capture map click events
        output = st_folium(m, width=700, height=500)

        # Handle map clicks
        if output and 'last_clicked' in output and output['last_clicked'] is not None:
            lat = output['last_clicked']['lat']
            lon = output['last_clicked']['lng']
            if attacker_type == 'Group of Drones':
                # Add multiple drones (e.g., 4 drones)
                num_drones_in_group = 4
                for _ in range(num_drones_in_group):
                    offset_lon = lon + np.random.uniform(-0.01, 0.01)
                    offset_lat = lat + np.random.uniform(-0.01, 0.01)
                    st.session_state['attackers'].append({
                        'type': 'Drone',
                        'position': [offset_lon, offset_lat],
                        'radar_cross_section': st.sidebar.slider("Radar Cross Section (m²)", 0.1, 10.0, 1.0),
                        'speed': st.sidebar.slider("Speed (m/s)", 100, 500, 250),
                        'altitude': st.sidebar.slider("Altitude (meters)", 100, 10000, 1000),
                        'guidance': st.sidebar.selectbox("Guidance Type", ['GPS', 'Inertial', 'Terrain Following'])
                    })
                st.success(f"Group of {num_drones_in_group} Drones added near ({lat:.5f}, {lon:.5f})")
            else:
                st.session_state['attackers'].append({
                    'type': attacker_type,
                    'position': [lon, lat],
                    'radar_cross_section': st.sidebar.slider("Radar Cross Section (m²)", 0.1, 10.0, 1.0),
                    'speed': st.sidebar.slider("Speed (m/s)", 100, 500, 250),
                    'altitude': st.sidebar.slider("Altitude (meters)", 100, 10000, 1000),
                    'guidance': st.sidebar.selectbox("Guidance Type", ['GPS', 'Inertial', 'Terrain Following'])
                })
                st.success(f"{attacker_type} added at ({lat:.5f}, {lon:.5f})")
            # Clear last clicked to prevent duplicate entries
            output['last_clicked'] = None

    with setup_tab2:
        st.subheader("Add Defenders")
        defender_type = st.selectbox("Select Defender Type", list(DEFENDER_TYPES.keys()))
        st.info("Select a point on the map to add a defender.")

        # Initialize map centered on Donetsk
        donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
        m = folium.Map(location=donetsk_coords, zoom_start=12)

        # Add existing defender markers
        for defender in st.session_state['defenders']:
            folium.Marker(
                location=[defender['position'][1], defender['position'][0]],
                icon=folium.Icon(color='red', icon='shield', prefix='fa'),
                popup=f"Defender: {defender['type']}"
            ).add_to(m)

        # Capture map click events
        output = st_folium(m, width=700, height=500)

        # Handle map clicks
        if output and 'last_clicked' in output and output['last_clicked'] is not None:
            lat = output['last_clicked']['lat']
            lon = output['last_clicked']['lng']
            st.session_state['defenders'].append({
                'type': defender_type,
                'position': [lon, lat],
                'missile_number': st.sidebar.number_input(f"Missile Number for {defender_type}", min_value=1, max_value=8, value=8),
                'max_targets': st.sidebar.number_input(f"Max Targets to Track for {defender_type}", min_value=1, max_value=10, value=2)
            })
            st.success(f"{defender_type} added at ({lat:.5f}, {lon:.5f})")
            # Clear last clicked to prevent duplicate entries
            output['last_clicked'] = None

    with setup_tab3:
        st.subheader("Set Target")
        st.write("Select a point on the map to set the target location for attackers.")

        # Initialize map centered on Donetsk
        donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
        m = folium.Map(location=donetsk_coords, zoom_start=12)

        # Add existing target marker
        if st.session_state['target_position']:
            folium.Marker(
                location=[st.session_state['target_position'][1], st.session_state['target_position'][0]],
                icon=folium.Icon(color='green', icon='star', prefix='fa'),
                popup="Target"
            ).add_to(m)

        # Capture map click events
        output = st_folium(m, width=700, height=500)

        # Handle map clicks
        if output and 'last_clicked' in output and output['last_clicked'] is not None:
            lat = output['last_clicked']['lat']
            lon = output['last_clicked']['lng']
            st.session_state['target_position'] = [lon, lat]
            st.success(f"Target set at ({lat:.5f}, {lon:.5f})")
            # Clear last clicked to prevent duplicate entries
            output['last_clicked'] = None

    st.markdown("---")

    # Display current entities
    st.subheader("Current Entities")
    st.write(f"**Total Attackers**: {len(st.session_state['attackers'])}")
    st.write(f"**Total Defenders**: {len(st.session_state['defenders'])}")
    st.write(f"**Target Set**: {'Yes' if st.session_state['target_position'] else 'No'}")

    # Start Simulation button
    if st.button("Start Simulation"):
        if len(st.session_state['attackers']) == 0 or len(st.session_state['defenders']) == 0 or not st.session_state['target_position']:
            st.error("Please add at least one Attacker, one Defender, and set the Target on the map.")
        else:
            st.session_state['run_simulation'] = True
            st.success("Simulation started!")
    else:
        st.session_state['run_simulation'] = False

    # Clear all markers
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
            params=st.session_state['params'],
            attackers=st.session_state['attackers'],
            defenders=st.session_state['defenders'],
            target_position=st.session_state['target_position'],
            log_area=log_area
        )

        # Display frames with video-like behavior
        video_placeholder = st.empty()
        for fig in frames:
            video_placeholder.pyplot(fig)
            time.sleep(0.1)  # Control the speed of frame display

        # Clear simulation state after completion
        st.session_state['attackers'] = []
        st.session_state['defenders'] = []
        st.session_state['target_position'] = None
        st.session_state['run_simulation'] = False

    else:
        st.write("Please set the parameters and starting positions in the 'Simulation Setup' tab.")