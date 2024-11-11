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
    if 'current_action' not in st.session_state:
        st.session_state['current_action'] = None
    if 'selected_attacker_type' not in st.session_state:
        st.session_state['selected_attacker_type'] = None
    if 'selected_defender_type' not in st.session_state:
        st.session_state['selected_defender_type'] = None
    if 'map_counter' not in st.session_state:
        st.session_state['map_counter'] = 0

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
def run_simulation(params, attackers, defenders, target_position, log_area, plot_placeholder):
    """
    Execute the air defense simulation.

    Parameters:
        params (dict): Simulation parameters.
        attackers (list): List of attacker dictionaries.
        defenders (list): List of defender dictionaries.
        target_position (list): [longitude, latitude] of the target.
        log_area (Streamlit container): Container to display logs.
        plot_placeholder (Streamlit container): Placeholder to display simulation frames.

    Returns:
        None
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

    current_time = 0  # Initialize simulation time

    for step in range(num_steps):
        fig, ax = plt.subplots(figsize=(10, 10))  # Fixed figure size
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
           
