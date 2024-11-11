import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import contextily as ctx
import folium
from streamlit_folium import st_folium
import logging
from datetime import datetime

# -------------------- Logging Configuration --------------------
# Configure logging to write logs to 'simulation.log' with timestamps
logging.basicConfig(
    filename='simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# -------------------- Defender and Attacker Definitions --------------------
# Define defender types with their parameters
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

# Define attacker types with their parameters
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

# -------------------- Session State Initialization --------------------
def initialize_session_state():
    """
    Initialize session state variables for attackers, defenders, and target.
    """
    if 'attackers' not in st.session_state:
        st.session_state['attackers'] = []
    if 'defenders' not in st.session_state:
        st.session_state['defenders'] = []
    if 'target_position' not in st.session_state:
        st.session_state['target_position'] = None
    if 'run_simulation' not in st.session_state:
        st.session_state['run_simulation'] = False
    if 'attackers_map_counter' not in st.session_state:
        st.session_state['attackers_map_counter'] = 0
    if 'defenders_map_counter' not in st.session_state:
        st.session_state['defenders_map_counter'] = 0

# -------------------- Entity Management Functions --------------------
def add_attacker(attacker_type, position):
    """
    Add an attacker to the session state.

    Parameters:
        attacker_type (str): Type of the attacker ('Drone' or 'Group of Drones').
        position (list): [longitude, latitude] coordinates.
    """
    if attacker_type == 'Drone':
        attacker = {
            'type': 'Drone',
            'position': position,
            'radar_cross_section': ATTACKER_TYPES['Drone']['radar_cross_section'],
            'speed': ATTACKER_TYPES['Drone']['default_speed'],
            'altitude': ATTACKER_TYPES['Drone']['min_altitude'],
            'guidance': ATTACKER_TYPES['Drone']['default_guidance']
        }
        st.session_state['attackers'].append(attacker)
        logger.info(f"Added Drone at {position}.")
    elif attacker_type == 'Group of Drones':
        num_drones = ATTACKER_TYPES['Group of Drones']['number_of_drones']
        for _ in range(num_drones):
            attacker = {
                'type': 'Drone',
                'position': position.copy(),  # Copy to avoid reference issues
                'radar_cross_section': ATTACKER_TYPES['Group of Drones']['radar_cross_section'],
                'speed': ATTACKER_TYPES['Group of Drones']['default_speed'],
                'altitude': ATTACKER_TYPES['Group of Drones']['min_altitude'],
                'guidance': ATTACKER_TYPES['Group of Drones']['default_guidance']
            }
            st.session_state['attackers'].append(attacker)
        logger.info(f"Added Group of {num_drones} Drones at {position}.")

def add_defender(defender_type, position):
    """
    Add a defender to the session state.

    Parameters:
        defender_type (str): Type of the defender (e.g., 'IRIS-T').
        position (list): [longitude, latitude] coordinates.
    """
    defender = {
        'type': defender_type,
        'position': position,
        'range': DEFENDER_TYPES[defender_type]['range'] / 111000,  # Convert meters to degrees
        'missile_number': DEFENDER_TYPES[defender_type]['default_missile_number'],
        'max_targets': DEFENDER_TYPES[defender_type]['max_targets']
    }
    st.session_state['defenders'].append(defender)
    logger.info(f"Added Defender '{defender_type}' at {position}.")

def update_attacker(idx, updated_params):
    """
    Update the parameters of an existing attacker.

    Parameters:
        idx (int): Index of the attacker in the session state list.
        updated_params (dict): Dictionary of updated parameters.
    """
    st.session_state['attackers'][idx].update(updated_params)
    logger.info(f"Updated Attacker {idx + 1}: {updated_params}.")

def update_defender(idx, updated_params):
    """
    Update the parameters of an existing defender.

    Parameters:
        idx (int): Index of the defender in the session state list.
        updated_params (dict): Dictionary of updated parameters.
    """
    st.session_state['defenders'][idx].update(updated_params)
    logger.info(f"Updated Defender {idx + 1}: {updated_params}.")

# -------------------- Simulation Function --------------------
def run_simulation(params, attackers, defenders, target_position, log_area):
    """
    Execute the air defense simulation.

    Parameters:
        params (dict): Simulation parameters.
        attackers (list): List of attacker dictionaries.
        defenders (list): List of defender dictionaries.
        target_position (list): [longitude, latitude] of the target.
        log_area (Streamlit container): Container to display logs.

    Returns:
        frames (list): List of matplotlib figures representing each simulation step.
    """
    # Unpack simulation parameters
    time_step = params['time_step']
    num_steps = params['num_steps']
    avoidance_radius = params['avoidance_radius'] / 111000  # Convert meters to degrees

    # Initialize attacker properties
    attacker_positions = np.array([attacker['position'] for attacker in attackers])
    attacker_speeds = [attacker['speed'] for attacker in attackers]
    attacker_active = [True] * len(attackers)
    attacker_paths = [[] for _ in attackers]
    attacker_types = [attacker['type'] for attacker in attackers]

    # Initialize defender properties
    defender_positions = np.array([defender['position'] for defender in defenders])
    defender_ranges = [defender['range'] for defender in defenders]
    defender_missile_number = [defender['missile_number'] for defender in defenders]
    defender_max_targets = [defender['max_targets'] for defender in defenders]
    defender_types_list = [defender['type'] for defender in defenders]

    missiles = []  # List to store active missiles
    interception_points = []  # List to store interception points

    # Initialize logs
    log_messages = []

    frames = []
    current_time = 0  # Initialize simulation time

    for step in range(num_steps):
        fig, ax = plt.subplots(figsize=(10, 10))
        # Set map bounds around Donetsk
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
                missiles_to_remove.append(missile)
                if missile['attacker_index'] is not None and attacker_active[missile['attacker_index']]:
                    attacker_active[missile['attacker_index']] = False
                    interception_points.append(attacker_positions[missile['attacker_index']].copy())
                    log_msg = (f"Time {current_time}s: {defender_types_list[missile['defender_index']]} "
                               f"{missile['defender_index'] + 1} intercepted "
                               f"{attacker_types[missile['attacker_index']]} "
                               f"{missile['attacker_index'] + 1}.")
                    logger.info(log_msg)
                    log_messages.append(log_msg)
            else:
                # Check if missile is out of range (simulate miss)
                missile_max_range = 200000 / 111000  # 200 km in degrees
                missile_travel_distance = np.linalg.norm(missile['position'] - missile['start_position'])
                if missile_travel_distance >= missile_max_range:
                    missiles_to_remove.append(missile)
                    log_msg = (f"Time {current_time}s: Missile from {defender_types_list[missile['defender_index']]} "
                               f"{missile['defender_index'] + 1} missed "
                               f"{attacker_types[missile['attacker_index']]} "
                               f"{missile['attacker_index'] + 1}.")
                    logger.info(log_msg)
                    log_messages.append(log_msg)

        # Remove missiles that have reached their targets or exceeded max range
        for missile in missiles_to_remove:
            missiles.remove(missile)

        # Launch new missiles from defenders
        for i, (defender_pos, defender_range) in enumerate(zip(defender_positions, defender_ranges)):
            if defender_missile_number[i] <= 0:
                continue  # No missiles left

            for j, (attacker_pos, active) in enumerate(zip(attacker_positions, attacker_active)):
                if not active:
                    continue
                distance = np.linalg.norm(attacker_pos - defender_pos)
                if distance <= defender_range:
                    # Launch missile towards attacker
                    missile_direction = (attacker_pos - defender_pos) / distance
                    missile_speed = 2000  # meters per time step
                    missile_velocity = missile_direction * missile_speed * time_step / 111000  # degrees per time step

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
                    log_msg = (f"Time {current_time}s: {defender_types_list[i]} "
                               f"{i + 1} launched missile at {attacker_types[j]} {j + 1}.")
                    logger.info(log_msg)
                    log_messages.append(log_msg)
                    break  # Defender cannot launch multiple missiles in the same step

        # Plot attacker paths
        for i, path in enumerate(attacker_paths):
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], color='blue', alpha=0.5, linestyle='--',
                        label='Attacker Path' if i == 0 else "")

        # Plot missile paths
        for missile in missiles:
            if len(missile['path']) > 1:
                path_array = np.array(missile['path'])
                ax.plot(path_array[:, 0], path_array[:, 1], color='orange', alpha=0.7, linestyle='-',
                        label='Missile Path' if missile == missiles[0] else "")

        # Plot interception points
        for point in interception_points:
            ax.plot(point[0], point[1], 'rx', markersize=10, label='Interception')

        # Plot active attackers
        for i, pos in enumerate(attacker_positions):
            if not attacker_active[i]:
                continue
            ax.plot(pos[0], pos[1], 'bo', markersize=5, label='Attacker' if i == 0 else "")

        # Plot defenders with their detection ranges
        for i, pos in enumerate(defender_positions):
            circle = Circle(
                pos,
                defender_ranges[i],
                color='red',
                alpha=0.1,
                transform=ax.transData._b
            )
            ax.add_patch(circle)
            ax.plot(pos[0], pos[1], 'ro', markersize=5, label='Defender' if i == 0 else "")

        # Add timestamp to the frame
        ax.text(0.01, 0.99, f'Time: {current_time}s', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Update logs
        active_attackers = np.sum(attacker_active)
        log_msg = f"Time {current_time}s: {active_attackers} attackers remaining."
        logger.info(log_msg)
        log_messages.append(log_msg)

        # Save the frame
        plt.title(f"Simulation Time: {current_time}s")
        frames.append(fig)
        plt.close()

        # Update the log area with the latest 10 messages
        log_area.text('\n'.join(log_messages[-10:]))
        logger.debug(f"Frame {step + 1}: Simulation at {current_time}s completed.")

        # Increment simulation time
        current_time += time_step

        # Terminate simulation if no attackers remain
        if active_attackers == 0:
            log_msg = f"All attackers have reached the target or been destroyed at time {current_time}s."
            logger.info(log_msg)
            log_messages.append(log_msg)
            break

    return frames

# -------------------- Map Rendering Functions --------------------
def render_target_map(target_position):
    """
    Render the Folium map for target placement.

    Parameters:
        target_position (list): [longitude, latitude] of the target.

    Returns:
        folium.Map: Rendered Folium map object.
    """
    donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
    m = folium.Map(location=donetsk_coords, zoom_start=12)
    if target_position:
        folium.Marker(
            location=[target_position[1], target_position[0]],
            icon=folium.Icon(color='green', icon='star', prefix='fa'),
            popup="Target"
        ).add_to(m)
    return m

def render_attackers_map(attacker_type):
    """
    Render the Folium map for attackers placement.

    Parameters:
        attacker_type (str): Type of the attacker being added.

    Returns:
        folium.Map: Rendered Folium map object.
    """
    donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
    m = folium.Map(location=donetsk_coords, zoom_start=12)
    # Add existing attackers of the selected type to the map
    for attacker in st.session_state['attackers']:
        if attacker['type'] == attacker_type:
            icon = 'plane' if attacker['type'] != 'Group of Drones' else 'fighter-jet'
            folium.Marker(
                location=[attacker['position'][1], attacker['position'][0]],
                icon=folium.Icon(color='blue', icon=icon, prefix='fa'),
                popup=f"{attacker['type']}"
            ).add_to(m)
    return m

def render_defenders_map(defender_type):
    """
    Render the Folium map for defenders placement.

    Parameters:
        defender_type (str): Type of the defender being added.

    Returns:
        folium.Map: Rendered Folium map object.
    """
    donetsk_coords = [48.0159, 37.8028]  # Latitude, Longitude
    m = folium.Map(location=donetsk_coords, zoom_start=12)
    # Add existing defenders of the selected type to the map
    for defender in st.session_state['defenders']:
        if defender['type'] == defender_type:
            folium.Marker(
                location=[defender['position'][1], defender['position'][0]],
                icon=folium.Icon(color='red', icon='shield', prefix='fa'),
                popup=f"{defender['type']}"
            ).add_to(m)
    return m

# -------------------- Streamlit Application --------------------
def main():
    """Main function to run the Streamlit application."""
    # Initialize session state
    initialize_session_state()

    # Set Streamlit page configuration
    st.set_page_config(page_title="Air Defense Simulation over Donetsk", layout="wide")

    # Application title
    st.title("Air Defense Simulation over Donetsk")

    # Define application tabs
    tab_setup, tab_attackers, tab_defenders, tab_results = st.tabs(
        ["Simulation Setup", "Attackers", "Defenders", "Simulation Results"]
    )

    # -------------------- Simulation Setup Tab --------------------
    with tab_setup:
        st.header("Simulation Setup")
        st.write("Configure general simulation parameters and set the target location.")

        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            avoidance_radius = st.slider(
                "Defender Avoidance Radius (meters)", 0, 5000, 1000, key='avoidance_radius'
            )
        with col2:
            num_steps = st.slider(
                "Number of Simulation Steps", 1, 100, 50, key='num_steps'
            )
        with col3:
            time_step = st.slider(
                "Time Step Duration (seconds)", 1, 10, 5, key='time_step'
            )

        # Consolidate simulation parameters
        params = {
            'avoidance_radius': avoidance_radius,
            'num_steps': num_steps,
            'time_step': time_step
        }

        st.subheader("Set Target Location")
        st.write("Use the map below to place the target location where attackers will navigate towards.")

        # Render target placement map
        target_map = render_target_map(st.session_state['target_position'])
        output_target = st_folium(target_map, width=700, height=500, key="target_map")

        # Handle target placement
        if output_target and 'last_clicked' in output_target and output_target['last_clicked'] is not None:
            lat = output_target['last_clicked']['lat']
            lon = output_target['last_clicked']['lng']
            st.session_state['target_position'] = [lon, lat]
            st.success(f"Target set at ({lat:.5f}, {lon:.5f})")
            logger.info(f"Target set at ({lon}, {lat}).")
            # Clear last clicked to prevent duplicate entries
            output_target['last_clicked'] = None

        # Display target status
        st.write(f"**Target Set**: {'Yes' if st.session_state['target_position'] else 'No'}")

        # Simulation control buttons
        if st.button("Start Simulation"):
            if not st.session_state['target_position']:
                st.error("Please set the target location on the map.")
                logger.error("Simulation start failed: Target location not set.")
            elif len(st.session_state['attackers']) == 0:
                st.error("Please add at least one Attacker in the 'Attackers' tab.")
                logger.error("Simulation start failed: No attackers added.")
            elif len(st.session_state['defenders']) == 0:
                st.error("Please add at least one Defender in the 'Defenders' tab.")
                logger.error("Simulation start failed: No defenders added.")
            else:
                st.session_state['params'] = params
                st.session_state['run_simulation'] = True
                logger.info("Simulation started.")
        else:
            st.session_state['run_simulation'] = False

        # Clear all simulation settings
        if st.button("Clear All Simulation Settings"):
            st.session_state['attackers'] = []
            st.session_state['defenders'] = []
            st.session_state['target_position'] = None
            st.session_state['run_simulation'] = False
            st.success("All simulation settings have been cleared.")
            logger.info("All simulation settings cleared.")

    # -------------------- Attackers Tab --------------------
    with tab_attackers:
        st.header("Attackers Management")
        st.write("Manage the list of attackers by adding or editing them.")

        # Radio buttons to select attacker type to add
        st.subheader("Add New Attacker")
        attacker_type = st.radio(
            "Select Attacker Type",
            list(ATTACKER_TYPES.keys()),
            key='select_attacker_type'
        )

        st.write("**Place Attacker on Map**")
        # Generate a unique key for the map using a counter
        map_key = f"attackers_map_{st.session_state['attackers_map_counter']}"
        st.session_state['attackers_map_counter'] += 1

        # Render attackers placement map
        attackers_map = render_attackers_map(attacker_type)
        output_attack = st_folium(attackers_map, width=700, height=400, key=map_key)

        # Handle attacker placement
        if output_attack and 'last_clicked' in output_attack and output_attack['last_clicked'] is not None:
            lat = output_attack['last_clicked']['lat']
            lon = output_attack['last_clicked']['lng']
            position = [lon, lat]
            add_attacker(attacker_type, position)
            st.success(f"{attacker_type} added at ({lat:.5f}, {lon:.5f}).")
            # Clear last clicked to prevent duplicate entries
            output_attack['last_clicked'] = None

        # Display list of attackers
        st.subheader("Existing Attackers")
        if len(st.session_state['attackers']) == 0:
            st.write("No attackers added yet.")
        else:
            for idx, attacker in enumerate(st.session_state['attackers']):
                with st.expander(f"Attacker {idx + 1}: {attacker['type']}"):
                    st.write(f"**Type**: {attacker['type']}")
                    st.write(f"**Position**: ({attacker['position'][1]:.5f}, {attacker['position'][0]:.5f})")

                    # Editable parameters based on attacker type
                    if attacker['type'] == 'Drone':
                        with st.form(key=f'attacker_form_{idx}'):
                            st.write("**Drone Parameters**")
                            radar_cs = st.number_input(
                                "Radar Cross Section (m²)",
                                min_value=0.01,
                                value=attacker['radar_cross_section'],
                                key=f'radar_cs_{idx}'
                            )
                            speed = st.number_input(
                                "Speed (meters per time step)",
                                min_value=100,
                                max_value=1000,
                                value=attacker['speed'],
                                key=f'speed_{idx}'
                            )
                            altitude = st.number_input(
                                "Altitude (meters)",
                                min_value=100,
                                max_value=10000,
                                value=attacker['altitude'],
                                key=f'altitude_{idx}'
                            )
                            guidance = st.selectbox(
                                "Guidance Type",
                                ATTACKER_TYPES['Drone']['guidance_types'],
                                index=ATTACKER_TYPES['Drone']['guidance_types'].index(attacker['guidance']),
                                key=f'guidance_{idx}'
                            )
                            submitted = st.form_submit_button("Update Drone")
                            if submitted:
                                updated_params = {
                                    'radar_cross_section': radar_cs,
                                    'speed': speed,
                                    'altitude': altitude,
                                    'guidance': guidance
                                }
                                update_attacker(idx, updated_params)
                                st.success(f"Drone {idx + 1} updated.")
                    elif attacker['type'] == 'Group of Drones':
                        with st.form(key=f'attacker_form_{idx}'):
                            st.write("**Group of Drones Parameters**")
                            radar_cs = st.number_input(
                                "Radar Cross Section per Drone (m²)",
                                min_value=0.01,
                                value=attacker['radar_cross_section'],
                                key=f'radar_cs_group_{idx}'
                            )
                            speed = st.number_input(
                                "Speed (meters per time step)",
                                min_value=100,
                                max_value=1000,
                                value=attacker['speed'],
                                key=f'speed_group_{idx}'
                            )
                            altitude = st.number_input(
                                "Altitude (meters)",
                                min_value=100,
                                max_value=10000,
                                value=attacker['altitude'],
                                key=f'altitude_group_{idx}'
                            )
                            guidance = st.selectbox(
                                "Guidance Type",
                                ATTACKER_TYPES['Group of Drones']['guidance_types'],
                                index=ATTACKER_TYPES['Group of Drones']['guidance_types'].index(attacker['guidance']),
                                key=f'guidance_group_{idx}'
                            )
                            number_of_drones = st.number_input(
                                "Number of Drones in Group",
                                min_value=1,
                                max_value=10,
                                value=ATTACKER_TYPES['Group of Drones']['number_of_drones'],
                                key=f'num_drones_group_{idx}'
                            )
                            submitted = st.form_submit_button("Update Group")
                            if submitted:
                                updated_params = {
                                    'radar_cross_section': radar_cs,
                                    'speed': speed,
                                    'altitude': altitude,
                                    'guidance': guidance,
                                    'number_of_drones': number_of_drones
                                }
                                update_attacker(idx, updated_params)
                                st.success(f"Group of Drones {idx + 1} updated.")

                    # Delete attacker button
                    if st.button(f"Delete Attacker {idx + 1}", key=f'delete_attacker_{idx}'):
                        del st.session_state['attackers'][idx]
                        st.success(f"Attacker {idx + 1} deleted.")
                        logger.info(f"Attacker {idx + 1} deleted.")

    # -------------------- Defenders Tab --------------------
    with tab_defenders:
        st.header("Defenders Management")
        st.write("Manage the list of defenders by adding or editing them.")

        # Radio buttons to select defender type to add
        st.subheader("Add New Defender")
        defender_type = st.radio(
            "Select Defender Type",
            list(DEFENDER_TYPES.keys()),
            key='select_defender_type'
        )

        st.write("**Place Defender on Map**")
        # Generate a unique key for the map using a counter
        map_key = f"defenders_map_{st.session_state['defenders_map_counter']}"
        st.session_state['defenders_map_counter'] += 1

        # Render defenders placement map
        defenders_map = render_defenders_map(defender_type)
        output_defender = st_folium(defenders_map, width=700, height=400, key=map_key)

        # Handle defender placement
        if output_defender and 'last_clicked' in output_defender and output_defender['last_clicked'] is not None:
            lat = output_defender['last_clicked']['lat']
            lon = output_defender['last_clicked']['lng']
            position = [lon, lat]
            add_defender(defender_type, position)
            st.success(f"{defender_type} Defender added at ({lat:.5f}, {lon:.5f}).")
            # Clear last clicked to prevent duplicate entries
            output_defender['last_clicked'] = None

        # Display list of defenders
        st.subheader("Existing Defenders")
        if len(st.session_state['defenders']) == 0:
            st.write("No defenders added yet.")
        else:
            for idx, defender in enumerate(st.session_state['defenders']):
                with st.expander(f"Defender {idx + 1}: {defender['type']}"):
                    st.write(f"**Type**: {defender['type']}")
                    st.write(f"**Position**: ({defender['position'][1]:.5f}, {defender['position'][0]:.5f})")
                    st.write(f"**Missiles Available**: {defender['missile_number']}/{DEFENDER_TYPES[defender['type']]['max_missile_number']}")

                    # Editable parameters based on defender type
                    if defender['type'] in DEFENDER_TYPES:
                        with st.form(key=f'defender_form_{idx}'):
                            st.write("**Defender Parameters**")
                            range_m = st.number_input(
                                "Range (meters)",
                                min_value=1000,
                                value=int(defender['range'] * 111000),
                                key=f'range_defender_{idx}'
                            )
                            max_targets = st.number_input(
                                "Max Targets to Track",
                                min_value=1,
                                value=defender['max_targets'],
                                key=f'max_targets_defender_{idx}'
                            )
                            missile_number = st.number_input(
                                "Missile Number",
                                min_value=0,
                                max_value=DEFENDER_TYPES[defender['type']]['max_missile_number'],
                                value=defender['missile_number'],
                                key=f'missile_number_defender_{idx}'
                            )
                            submitted = st.form_submit_button("Update Defender")
                            if submitted:
                                updated_params = {
                                    'range': range_m / 111000,  # Convert meters to degrees
                                    'max_targets': max_targets,
                                    'missile_number': missile_number
                                }
                                update_defender(idx, updated_params)
                                st.success(f"Defender {idx + 1} updated.")

                    # Delete defender button
                    if st.button(f"Delete Defender {idx + 1}", key=f'delete_defender_{idx}'):
                        del st.session_state['defenders'][idx]
                        st.success(f"Defender {idx + 1} deleted.")
                        logger.info(f"Defender {idx + 1} deleted.")

    # -------------------- Simulation Results Tab --------------------
    with tab_results:
        st.header("Simulation Results")
        st.write("Watch the simulation unfold with visualizations and logs.")

        if st.session_state['run_simulation']:
            # Placeholder for logs
            log_area = st.empty()

            # Run the simulation
            frames = run_simulation(
                st.session_state['params'],
                st.session_state['attackers'],
                st.session_state['defenders'],
                st.session_state['target_position'],
                log_area
            )

            # Display simulation frames
            for idx, fig in enumerate(frames):
                st.pyplot(fig)
                st.text(f"Frame {idx + 1}/{len(frames)} - Time {idx * st.session_state['params']['time_step']}s")
                time.sleep(0.5)  # Adjust speed as needed

            st.success("Simulation completed.")
            logger.info("Simulation completed.")

            # Reset simulation run flag
            st.session_state['run_simulation'] = False
        else:
            st.write("Please configure the simulation in the 'Simulation Setup', 'Attackers', and 'Defenders' tabs, then start the simulation.")

# -------------------- Run the Application --------------------
if __name__ == "__main__":
    main()
