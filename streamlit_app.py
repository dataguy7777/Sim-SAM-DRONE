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

# Define real ranges and default parameters for defenders
DEFENDER_TYPES = {
    'IRIS-T': {
        'type': 'SAM',
        'default_missile_number': 4,
        'max_missile_number': 8,
        'range': 40000,  # in meters
        'max_targets': 2,
    },
    'Hawk SAM': {
        'type': 'SAM',
        'default_missile_number': 6,
        'max_missile_number': 12,
        'range': 50000,
        'max_targets': 3,
    },
    'Patriot': {
        'type': 'SAM',
        'default_missile_number': 10,
        'max_missile_number': 20,
        'range': 160000,
        'max_targets': 5,
    },
    'Gepard': {
        'type': 'Anti-Aircraft Gun',
        'default_missile_number': 0,  # No missiles
        'max_missile_number': 0,
        'range': 5500,
        'max_targets': 1,
    }
}

# Define attacker types and default parameters
ATTACKER_TYPES = {
    'Drone': {
        'radar_cross_section': 0.1,  # in m²
        'default_speed': 250,  # meters per time step
        'min_altitude': 500,  # meters
        'max_altitude': 5000,  # meters
        'guidance_types': ['GPS', 'Inertial', 'Terrain Following'],
        'default_guidance': 'GPS'
    },
    'Group of Drones': {
        'radar_cross_section': 0.1,  # in m²
        'default_speed': 250,  # meters per time step
        'min_altitude': 500,  # meters
        'max_altitude': 5000,  # meters
        'guidance_types': ['GPS', 'Inertial', 'Terrain Following'],
        'default_guidance': 'GPS',
        'number_of_drones': 4  # Group size
    }
}

# Initialize session state
if 'attackers' not in st.session_state:
    st.session_state['attackers'] = []
if 'defenders' not in st.session_state:
    st.session_state['defenders'] = []
if 'target_position' not in st.session_state:
    st.session_state['target_position'] = None

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
    attacker_radar_cross = []
    attacker_altitudes = []
    attacker_guidance = []
    for attacker in attackers:
        attacker_positions.append(attacker['position'])
        attacker_speeds.append(ATTACKER_TYPES[attacker['type']]['default_speed'])
        attacker_active.append(True)
        attacker_paths.append([])
        attacker_types.append(attacker['type'])
        attacker_radar_cross.append(attacker['radar_cross_section'])
        attacker_altitudes.append(attacker['altitude'])
        attacker_guidance.append(attacker['guidance'])
    attacker_positions = np.array(attacker_positions)

    # Initialize defenders
    defender_positions = []
    defender_ranges = []
    defender_missile_number = []
    defender_max_targets = []
    defender_active = []
    defender_types = []
    for defender in defenders:
        defender_positions.append(defender['position'])
        defender_ranges.append(defender['range'] / 111000)  # Convert meters to degrees
        defender_missile_number.append(defender['missile_number'])
        defender_max_targets.append(defender['max_targets'])
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
                if distance <= defender_range and defender_missile_number[i] > 0:
                    # Launch missile towards attacker
                    missile_direction = (attacker_pos - defender_pos) / distance
                    missile_speed = 2000  # meters per time step
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
                    defender_missile_number[i] -= 1  # Decrement available missiles
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
                ax.plot(path_array[:, 0], path_array[:, 1], color='blue', alpha=0.5, linestyle='--', label='Attacker Path' if i == 0 else "")

        # Missile paths
        for missile in missiles:
            if len(missile['path']) > 1:
                path_array = np.array(missile['path'])
                ax.plot(path_array[:, 0], path_array[:, 1], color='orange', alpha=0.7, linestyle='-', label='Missile Path' if missile == missiles[0] else "")

        # Draw interception points
        for point in interception_points:
            ax.plot(point[0], point[1], 'rx', markersize=10, label='Interception')

        # Draw attackers
        for i in range(len(attacker_positions)):
            if not attacker_active[i]:
                continue
            attacker_pos = attacker_positions[i]
            # Draw attacker as a blue dot
            ax.plot(*attacker_pos, 'bo', markersize=5, label='Attacker' if i == 0 else "")

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
            ax.plot(*defender_pos, 'ro', markersize=5, label='Defender' if i == 0 else "")

        # Add timestamp to frame
        ax.text(0.01, 0.99, f'Time: {current_time}s', transform=ax.transAxes, fontsize=12,
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
st.set_page_config(page_title="Air Defense Simulation over Donetsk", layout="wide")
st.title("Air Defense Simulation over Donetsk")

# Define tabs
tab_setup, tab_attackers, tab_defenders, tab_results = st.tabs(["Simulation Setup", "Attackers", "Defenders", "Simulation Results"])

with tab_setup:
    st.header("Simulation Setup")
    st.write("Configure general simulation parameters and set the target location.")

    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        avoidance_radius = st.slider("Defender Avoidance Radius (meters)", 0, 5000, 1000)
    with col2:
        num_steps = st.slider("Number of Simulation Steps", 1, 100, 50)
    with col3:
        time_step = st.slider("Time Step Duration (seconds)", 1, 10, 5)

    params = {
        'avoidance_radius': avoidance_radius,
        'num_steps': num_steps,
        'time_step': time_step
    }

    st.subheader("Set Target Location")
    st.write("Use the map to place the target location where attackers will navigate towards.")

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

    # Handle map clicks for target
    if output and 'last_clicked' in output and output['last_clicked'] is not None:
        lat = output['last_clicked']['lat']
        lon = output['last_clicked']['lng']
        st.session_state['target_position'] = [lon, lat]
        st.success(f"Target set at ({lat:.5f}, {lon:.5f})")
        # Clear last clicked to prevent duplicate entries
        output['last_clicked'] = None

    # Display target status
    st.write(f"**Target Set**: {'Yes' if st.session_state['target_position'] else 'No'}")

    # Start Simulation button
    if st.button("Start Simulation"):
        if not st.session_state['target_position']:
            st.error("Please set the target location on the map.")
        elif len(st.session_state['attackers']) == 0:
            st.error("Please add at least one Attacker in the 'Attackers' tab.")
        elif len(st.session_state['defenders']) == 0:
            st.error("Please add at least one Defender in the 'Defenders' tab.")
        else:
            st.session_state['params'] = params
            st.session_state['run_simulation'] = True
    else:
        st.session_state['run_simulation'] = False

    # Clear all markers button
    if st.button("Clear All Simulation Settings"):
        st.session_state['attackers'] = []
        st.session_state['defenders'] = []
        st.session_state['target_position'] = None
        st.success("All simulation settings have been cleared.")

with tab_attackers:
    st.header("Attackers Management")
    st.write("Manage the list of attackers by adding or editing them.")

    # List of attackers
    for idx, attacker in enumerate(st.session_state['attackers']):
        with st.expander(f"Attacker {idx + 1}: {attacker['type']}"):
            st.write(f"**Type**: {attacker['type']}")
            st.write(f"**Position**: ({attacker['position'][1]:.5f}, {attacker['position'][0]:.5f})")

            # Editable parameters based on type
            if attacker['type'] == 'Drone':
                with st.form(key=f'attacker_form_{idx}'):
                    st.write("**Drone Parameters**")
                    radar_cs = st.number_input("Radar Cross Section (m²)", min_value=0.01, value=attacker['radar_cross_section'])
                    speed = st.number_input("Speed (meters per time step)", min_value=100, max_value=1000, value=attacker['speed'])
                    altitude = st.number_input("Altitude (meters)", min_value=100, max_value=10000, value=attacker['altitude'])
                    guidance = st.selectbox("Guidance Type", ATTACKER_TYPES['Drone']['guidance_types'], index=ATTACKER_TYPES['Drone']['guidance_types'].index(attacker['guidance']))
                    submitted = st.form_submit_button("Update Drone")
                    if submitted:
                        st.session_state['attackers'][idx]['radar_cross_section'] = radar_cs
                        st.session_state['attackers'][idx]['speed'] = speed
                        st.session_state['attackers'][idx]['altitude'] = altitude
                        st.session_state['attackers'][idx]['guidance'] = guidance
                        st.success(f"Drone {idx + 1} updated.")
            elif attacker['type'] == 'Group of Drones':
                with st.form(key=f'attacker_form_{idx}'):
                    st.write("**Group of Drones Parameters**")
                    radar_cs = st.number_input("Radar Cross Section per Drone (m²)", min_value=0.01, value=attacker['radar_cross_section'])
                    speed = st.number_input("Speed (meters per time step)", min_value=100, max_value=1000, value=attacker['speed'])
                    altitude = st.number_input("Altitude (meters)", min_value=100, max_value=10000, value=attacker['altitude'])
                    guidance = st.selectbox("Guidance Type", ATTACKER_TYPES['Group of Drones']['guidance_types'], index=ATTACKER_TYPES['Group of Drones']['guidance_types'].index(attacker['guidance']))
                    number_of_drones = st.number_input("Number of Drones in Group", min_value=1, max_value=10, value=ATTACKER_TYPES['Group of Drones']['number_of_drones'])
                    submitted = st.form_submit_button("Update Group")
                    if submitted:
                        st.session_state['attackers'][idx]['radar_cross_section'] = radar_cs
                        st.session_state['attackers'][idx]['speed'] = speed
                        st.session_state['attackers'][idx]['altitude'] = altitude
                        st.session_state['attackers'][idx]['guidance'] = guidance
                        st.session_state['attackers'][idx]['number_of_drones'] = number_of_drones
                        st.success(f"Group of Drones {idx + 1} updated.")

            # Delete attacker button
            if st.button(f"Delete Attacker {idx + 1}", key=f'delete_attacker_{idx}'):
                del st.session_state['attackers'][idx]
                st.success(f"Attacker {idx + 1} deleted.")

    st.subheader("Add New Attacker")
    with st.form(key='add_attacker_form'):
        attacker_type = st.selectbox("Select Attacker Type", list(ATTACKER_TYPES.keys()))
        if attacker_type in ['Drone', 'Group of Drones']:
            st.write("**Place Attacker on Map**")
            # Initialize map centered on Donetsk
            m_attack = folium.Map(location=donetsk_coords, zoom_start=12)

            # Add existing attackers of this type to the map
            for attacker in st.session_state['attackers']:
                if attacker['type'] == attacker_type:
                    folium.Marker(
                        location=[attacker['position'][1], attacker['position'][0]],
                        icon=folium.Icon(color='blue', icon='plane', prefix='fa'),
                        popup=f"{attacker['type']}"
                    ).add_to(m_attack)

            # Capture map click events
            output_attack = st_folium(m_attack, width=700, height=400)

            # Handle map clicks for adding attacker
            if output_attack and 'last_clicked' in output_attack and output_attack['last_clicked'] is not None:
                lat = output_attack['last_clicked']['lat']
                lon = output_attack['last_clicked']['lng']
                if attacker_type == 'Drone':
                    st.session_state['attackers'].append({
                        'type': 'Drone',
                        'position': [lon, lat],
                        'radar_cross_section': ATTACKER_TYPES['Drone']['radar_cross_section'],
                        'speed': ATTACKER_TYPES['Drone']['default_speed'],
                        'altitude': ATTACKER_TYPES['Drone']['min_altitude'],
                        'guidance': ATTACKER_TYPES['Drone']['default_guidance']
                    })
                    st.success(f"Drone added at ({lat:.5f}, {lon:.5f})")
                elif attacker_type == 'Group of Drones':
                    num = ATTACKER_TYPES['Group of Drones']['number_of_drones']
                    for _ in range(num):
                        st.session_state['attackers'].append({
                            'type': 'Drone',
                            'position': [lon, lat],
                            'radar_cross_section': ATTACKER_TYPES['Group of Drones']['radar_cross_section'],
                            'speed': ATTACKER_TYPES['Group of Drones']['default_speed'],
                            'altitude': ATTACKER_TYPES['Group of Drones']['min_altitude'],
                            'guidance': ATTACKER_TYPES['Group of Drones']['default_guidance']
                        })
                    st.success(f"Group of {num} Drones added at ({lat:.5f}, {lon:.5f})")
                # Clear last clicked to prevent duplicate entries
                output_attack['last_clicked'] = None

        submitted_attacker = st.form_submit_button("Add Attacker")
        if submitted_attacker:
            st.success("Click on the map to place the attacker.")

with tab_defenders:
    st.header("Defenders Management")
    st.write("Manage the list of defenders by adding or editing them.")

    # List of defenders
    for idx, defender in enumerate(st.session_state['defenders']):
        with st.expander(f"Defender {idx + 1}: {defender['type']}"):
            st.write(f"**Type**: {defender['type']}")
            st.write(f"**Position**: ({defender['position'][1]:.5f}, {defender['position'][0]:.5f})")
            st.write(f"**Missiles Available**: {defender['missile_number']}/{DEFENDER_TYPES[defender['type']]['max_missile_number']}")

            # Editable parameters based on type
            if defender['type'] == 'IRIS-T' or defender['type'] == 'Hawk SAM' or defender['type'] == 'Patriot' or defender['type'] == 'Gepard':
                with st.form(key=f'defender_form_{idx}'):
                    st.write("**Defender Parameters**")
                    range_m = st.number_input("Range (meters)", min_value=1000, value=defender['range'] * 111000)
                    max_targets = st.number_input("Max Targets to Track", min_value=1, value=defender['max_targets'])
                    missile_number = st.number_input("Missile Number", min_value=0, max_value=DEFENDER_TYPES[defender['type']]['max_missile_number'], value=defender['missile_number'])
                    submitted = st.form_submit_button("Update Defender")
                    if submitted:
                        st.session_state['defenders'][idx]['range'] = range_m / 111000  # Convert back to degrees
                        st.session_state['defenders'][idx]['max_targets'] = max_targets
                        st.session_state['defenders'][idx]['missile_number'] = missile_number
                        st.success(f"Defender {idx + 1} updated.")

            # Delete defender button
            if st.button(f"Delete Defender {idx + 1}", key=f'delete_defender_{idx}'):
                del st.session_state['defenders'][idx]
                st.success(f"Defender {idx + 1} deleted.")

    st.subheader("Add New Defender")
    with st.form(key='add_defender_form'):
        defender_type = st.selectbox("Select Defender Type", list(DEFENDER_TYPES.keys()))
        st.write("**Place Defender on Map**")
        # Initialize map centered on Donetsk
        m_def = folium.Map(location=donetsk_coords, zoom_start=12)

        # Add existing defenders of this type to the map
        for defender in st.session_state['defenders']:
            if defender['type'] == defender_type:
                folium.Marker(
                    location=[defender['position'][1], defender['position'][0]],
                    icon=folium.Icon(color='red', icon='shield', prefix='fa'),
                    popup=f"{defender['type']}"
                ).add_to(m_def)

        # Capture map click events
        output_def = st_folium(m_def, width=700, height=400)

        # Handle map clicks for adding defender
        if output_def and 'last_clicked' in output_def and output_def['last_clicked'] is not None:
            lat = output_def['last_clicked']['lat']
            lon = output_def['last_clicked']['lng']
            st.session_state['defenders'].append({
                'type': defender_type,
                'position': [lon, lat],
                'range': DEFENDER_TYPES[defender_type]['range'] / 111000,  # Convert meters to degrees
                'missile_number': DEFENDER_TYPES[defender_type]['default_missile_number'],
                'max_targets': DEFENDER_TYPES[defender_type]['max_targets']
            })
            st.success(f"{defender_type} Defender added at ({lat:.5f}, {lon:.5f})")
            # Clear last clicked to prevent duplicate entries
            output_def['last_clicked'] = None

        submitted_defender = st.form_submit_button("Add Defender")
        if submitted_defender:
            st.success("Click on the map to place the defender.")

with tab_results:
    st.header("Simulation Results")
    st.write("Watch the simulation unfold with visualizations and logs.")

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

        # Display frames as a video-like animation
        for idx, fig in enumerate(frames):
            st.pyplot(fig)
            st.text(f"Frame {idx + 1}/{len(frames)} - Time {idx * st.session_state['params']['time_step']}s")
            time.sleep(0.5)  # Control the speed of frame display

        st.success("Simulation completed.")

        # Clear simulation run flag
        st.session_state['run_simulation'] = False

    else:
        st.write("Please configure the simulation in the 'Simulation Setup', 'Attackers', and 'Defenders' tabs, then start the simulation.")

