#!/usr/bin/env python3

import sys
import time
import math
import random
from pdu_generator import PDUGenerator
import pandas as pd

# Import OpenDIS classes
from opendis.dis7 import *
from opendis.DataOutputStream import DataOutputStream

# ============================================================================
# Scenario Functions
# ============================================================================

def run_static_vehicles_scenario(generator):
    """
    Scenario 1: Static Vehicles
    Creates three static ground vehicles at different positions
    """
    print("\n=== STATIC VEHICLES SCENARIO ===")
    print("Creating 3 static ground vehicles...")
    
    # Define positions
    positions = [
        (0.0, 0.0, 0.0),          # Origin
        (500.0, 0.0, 0.0),        # 500m East
        (0.0, 500.0, 0.0)         # 500m North
    ]
    
    # Create and send PDUs
    for i, position in enumerate(positions):
        entity_id = i + 1
        
        # Create tank PDU
        pdu = generator.create_static_tank_pdu(
            entity_id=entity_id,
            position=position
        )
        
        # Send PDU
        generator.send_pdu(pdu)
        
        print(f"Created Tank {entity_id} at position {position}")
        
        # Brief delay between transmissions
        time.sleep(0.5)
    
    print("All vehicle PDUs sent successfully")

def run_moving_aircraft_scenario(generator):
    """
    Scenario 2: Moving Aircraft
    Creates an aircraft that moves along a path
    """
    print("\n=== MOVING AIRCRAFT SCENARIO ===")
    
    # Aircraft properties
    entity_id = 10
    position = (0.0, 0.0, 1000.0)
    velocity = (100.0, 0.0, 0.0)  # Moving east at 100 m/s
    
    print(f"Creating aircraft (ID: {entity_id}) at position {position} with velocity {velocity}")
    
    # Send initial PDU
    initial_pdu = generator.create_moving_aircraft_pdu(
        entity_id=entity_id,
        position=position,
        velocity=velocity
    )
    generator.send_pdu(initial_pdu)
    
    # Simulate aircraft movement by sending updated PDUs
    print("Aircraft moving... (sending 10 updates)")
    
    for i in range(10):
        # Wait before next update
        time.sleep(1)
        
        # Update position based on velocity (simple linear movement)
        position = (
            position[0] + velocity[0],
            position[1] + velocity[1],
            position[2] + velocity[2]
        )
        
        # Create new PDU with updated position
        update_pdu = generator.create_moving_aircraft_pdu(
            entity_id=entity_id,
            position=position,
            velocity=velocity
        )
        
        # Send update
        generator.send_pdu(update_pdu)
        print(f"Update {i+1}/10: Aircraft at position {position}")
    
    print("Aircraft movement simulation completed")

def run_air_combat_scenario(generator):
    """
    Scenario 3: Air Combat
    Simulates an aircraft firing at a ground target
    """
    print("\n=== AIR COMBAT SCENARIO ===")
    
    # Step 1: Create a tank (target)
    print("Step 1: Creating ground target (tank)")
    tank_id = 20
    tank_position = (1000.0, 1000.0, 0.0)
    
    tank_pdu = generator.create_static_tank_pdu(
        entity_id=tank_id,
        position=tank_position
    )
    generator.send_pdu(tank_pdu)
    print(f"Created tank (ID: {tank_id}) at position {tank_position}")
    
    # Step 2: Create an aircraft
    print("\nStep 2: Creating aircraft")
    aircraft_id = 21
    aircraft_position = (0.0, 0.0, 2000.0)
    aircraft_velocity = (200.0, 200.0, 0.0)  # Moving toward tank
    
    aircraft_pdu = generator.create_moving_aircraft_pdu(
        entity_id=aircraft_id,
        position=aircraft_position,
        velocity=aircraft_velocity
    )
    generator.send_pdu(aircraft_pdu)
    print(f"Created aircraft (ID: {aircraft_id}) at position {aircraft_position}")
    
    # Step 3: Aircraft approaches target
    print("\nStep 3: Aircraft approaching target...")
    
    for i in range(3):
        # Wait a bit
        time.sleep(1)
        
        # Update aircraft position (moving toward target)
        aircraft_position = (
            aircraft_position[0] + aircraft_velocity[0],
            aircraft_position[1] + aircraft_velocity[1],
            aircraft_position[2] + aircraft_velocity[2]
        )
        
        # Send updated aircraft PDU
        aircraft_pdu = generator.create_moving_aircraft_pdu(
            entity_id=aircraft_id,
            position=aircraft_position,
            velocity=aircraft_velocity
        )
        generator.send_pdu(aircraft_pdu)
        print(f"Update {i+1}/3: Aircraft at position {aircraft_position}")
    
    # Step 4: Aircraft fires at target
    print("\nStep 4: Aircraft firing at target")
    munition_id = 22
    fire_position = aircraft_position
    munition_velocity = (300.0, 300.0, -50.0)  # Missile velocity
    
    fire_pdu = generator.create_fire_pdu(
        firing_entity_id=aircraft_id,
        target_entity_id=tank_id,
        munition_entity_id=munition_id,
        fire_position=fire_position,
        velocity=munition_velocity
    )
    generator.send_pdu(fire_pdu)
    print(f"Aircraft (ID: {aircraft_id}) fired munition (ID: {munition_id}) at tank (ID: {tank_id})")
    
    # Step 5: Missile flight
    print("\nStep 5: Missile in flight...")
    time.sleep(3)  # Wait for missile flight
    
    # Step 6: Impact/detonation
    print("\nStep 6: Missile impact")
    detonation_pdu = generator.create_detonation_pdu(
        firing_entity_id=aircraft_id,
        target_entity_id=tank_id,
        munition_entity_id=munition_id,
        detonation_position=tank_position,
        velocity=(50.0, 50.0, -10.0),
        detonation_result=1  # Entity impact
    )
    generator.send_pdu(detonation_pdu)
    print(f"Munition (ID: {munition_id}) detonated on tank (ID: {tank_id})")
    
    # Step 7: Aircraft continues on
    print("\nStep 7: Aircraft continuing on flight path")
    
    for i in range(2):
        # Wait a bit
        time.sleep(1)
        
        # Update aircraft position (continuing on path)
        aircraft_position = (
            aircraft_position[0] + aircraft_velocity[0],
            aircraft_position[1] + aircraft_velocity[1],
            aircraft_position[2] + aircraft_velocity[2]
        )
        
        # Send updated aircraft PDU
        aircraft_pdu = generator.create_moving_aircraft_pdu(
            entity_id=aircraft_id,
            position=aircraft_position,
            velocity=aircraft_velocity
        )
        generator.send_pdu(aircraft_pdu)
        print(f"Update {i+1}/2: Aircraft at position {aircraft_position}")
    
    print("Air combat scenario completed")

def run_collision_scenario(generator):
    """
    Scenario 4: Collision
    Simulates two entities on a collision course
    """
    print("\n=== COLLISION SCENARIO ===")
    
    # Create two entities on a collision course
    print("Step 1: Creating two entities on collision course")
    
    # Entity 1
    entity1_id = 30
    entity1_position = (0.0, 100.0, 0.0)
    entity1_velocity = (50.0, 0.0, 0.0)  # Moving east
    
    entity1_pdu = generator.create_moving_aircraft_pdu(
        entity_id=entity1_id,
        position=entity1_position,
        velocity=entity1_velocity
    )
    generator.send_pdu(entity1_pdu)
    print(f"Created entity 1 (ID: {entity1_id}) at position {entity1_position}")
    
    # Entity 2
    entity2_id = 31
    entity2_position = (500.0, 100.0, 0.0)
    entity2_velocity = (-50.0, 0.0, 0.0)  # Moving west
    
    entity2_pdu = generator.create_moving_aircraft_pdu(
        entity_id=entity2_id,
        position=entity2_position,
        velocity=entity2_velocity
    )
    generator.send_pdu(entity2_pdu)
    print(f"Created entity 2 (ID: {entity2_id}) at position {entity2_position}")
    
    # Simulate movement and detect collision
    print("\nStep 2: Entities moving toward each other")
    collision_detected = False
    
    for i in range(5):
        # Wait a bit
        time.sleep(1)
        
        # Update positions
        entity1_position = (
            entity1_position[0] + entity1_velocity[0],
            entity1_position[1] + entity1_velocity[1],
            entity1_position[2] + entity1_velocity[2]
        )
        
        entity2_position = (
            entity2_position[0] + entity2_velocity[0],
            entity2_position[1] + entity2_velocity[1],
            entity2_position[2] + entity2_velocity[2]
        )
        
        # Send updated PDUs
        entity1_pdu = generator.create_moving_aircraft_pdu(
            entity_id=entity1_id,
            position=entity1_position,
            velocity=entity1_velocity
        )
        generator.send_pdu(entity1_pdu)
        
        entity2_pdu = generator.create_moving_aircraft_pdu(
            entity_id=entity2_id,
            position=entity2_position,
            velocity=entity2_velocity
        )
        generator.send_pdu(entity2_pdu)
        
        print(f"Update {i+1}/5: Entities moving")
        print(f"  Entity 1: {entity1_position}")
        print(f"  Entity 2: {entity2_position}")
        
        # Check for collision (very basic - just check if they're close)
        distance = math.sqrt(
            (entity1_position[0] - entity2_position[0])**2 +
            (entity1_position[1] - entity2_position[1])**2 +
            (entity1_position[2] - entity2_position[2])**2
        )
        
        if distance < 50.0 and not collision_detected:
            collision_detected = True
            
            # Calculate collision point (halfway between entities)
            collision_position = (
                (entity1_position[0] + entity2_position[0]) / 2,
                (entity1_position[1] + entity2_position[1]) / 2,
                (entity1_position[2] + entity2_position[2]) / 2
            )
            
            print("\nStep 3: Collision detected!")
            print(f"Collision at position {collision_position}")
            
            # Create and send collision PDU
            collision_pdu = generator.create_collision_pdu(
                entity_id=entity1_id,
                collided_entity_id=entity2_id,
                collision_position=collision_position,
                velocity=entity1_velocity,
                mass=15000.0,
                collision_type=1  # Inelastic collision
            )
            generator.send_pdu(collision_pdu)
            print(f"Sent collision PDU for entities {entity1_id} and {entity2_id}")
    
    if not collision_detected:
        print("\nNo collision detected within simulation steps")
    
    print("Collision scenario completed")

def run_formation_flight_scenario(generator):
    """
    Scenario 5: Formation Flight
    Simulates multiple aircraft flying in formation
    """
    print("\n=== FORMATION FLIGHT SCENARIO ===")
    
    # Create leader aircraft
    print("Step 1: Creating leader aircraft")
    leader_id = 40
    leader_position = (0.0, 0.0, 1000.0)
    leader_velocity = (100.0, 0.0, 0.0)  # Moving east
    
    leader_pdu = generator.create_moving_aircraft_pdu(
        entity_id=leader_id,
        position=leader_position,
        velocity=leader_velocity
    )
    generator.send_pdu(leader_pdu)
    print(f"Created leader (ID: {leader_id}) at position {leader_position}")
    
    # Create wingmen with offsets from leader
    print("\nStep 2: Creating wingmen")
    wingmen_offsets = [
        (-100.0, -100.0, 0.0),    # Left wing
        (100.0, -100.0, 0.0),     # Right wing
        (0.0, -200.0, -50.0)      # Rear
    ]
    
    wingmen_ids = [41, 42, 43]
    wingmen_positions = []
    
    for i, (offset, wingman_id) in enumerate(zip(wingmen_offsets, wingmen_ids)):
        # Calculate wingman position relative to leader
        wingman_position = (
            leader_position[0] + offset[0],
            leader_position[1] + offset[1],
            leader_position[2] + offset[2]
        )
        wingmen_positions.append(wingman_position)
        
        # Create and send wingman PDU
        wingman_pdu = generator.create_moving_aircraft_pdu(
            entity_id=wingman_id,
            position=wingman_position,
            velocity=leader_velocity  # Same velocity as leader
        )
        generator.send_pdu(wingman_pdu)
        print(f"Created wingman {i+1} (ID: {wingman_id}) at position {wingman_position}")
    
    # Simulate formation flying
    print("\nStep 3: Formation in flight")
    
    for i in range(10):
        # Wait a bit
        time.sleep(1)
        
        # Update leader position
        leader_position = (
            leader_position[0] + leader_velocity[0],
            leader_position[1] + leader_velocity[1],
            leader_position[2] + leader_velocity[2]
        )
        
        # Create and send updated leader PDU
        leader_pdu = generator.create_moving_aircraft_pdu(
            entity_id=leader_id,
            position=leader_position,
            velocity=leader_velocity
        )
        generator.send_pdu(leader_pdu)
        
        # Update wingmen positions and send PDUs
        for j, (offset, wingman_id) in enumerate(zip(wingmen_offsets, wingmen_ids)):
            # Calculate wingman position relative to leader
            wingman_position = (
                leader_position[0] + offset[0],
                leader_position[1] + offset[1],
                leader_position[2] + offset[2]
            )
            
            # Create and send wingman PDU
            wingman_pdu = generator.create_moving_aircraft_pdu(
                entity_id=wingman_id,
                position=wingman_position,
                velocity=leader_velocity  # Same velocity as leader
            )
            generator.send_pdu(wingman_pdu)
        
        print(f"Update {i+1}/10: Formation at position {leader_position}")
    
    print("Formation flight scenario completed")

def run_terrain_following_scenario(generator):
    """
    Scenario 6: Terrain Following
    Simulates an aircraft flying low and following terrain
    """
    print("\n=== TERRAIN FOLLOWING SCENARIO ===")
    
    # Create aircraft
    entity_id = 50
    position = (0.0, 0.0, 200.0)  # Starting position, low altitude
    velocity = (80.0, 0.0, 0.0)   # Moving east
    
    # Simple terrain model (x coordinate -> altitude)
    # This represents hills and valleys the aircraft will follow
    terrain_profile = {
        0: 0,          # Start altitude
        500: 100,      # Hill at 500m east
        1000: 50,      # Valley after the hill
        1500: 200,     # Bigger hill
        2000: 30,      # Return to lower elevation
        2500: 0        # Back to starting elevation
    }
    
    # Send initial PDU
    aircraft_pdu = generator.create_moving_aircraft_pdu(
        entity_id=entity_id,
        position=position,
        velocity=velocity
    )
    generator.send_pdu(aircraft_pdu)
    print(f"Created terrain-following aircraft (ID: {entity_id}) at position {position}")
    
    # Simulate aircraft following terrain
    print("\nAircraft following terrain...")
    
    current_x = 0
    
    for i in range(30):
        # Wait a bit
        time.sleep(0.5)
        
        # Update x position based on velocity
        current_x += velocity[0] * 0.5  # 0.5s * velocity
        
        # Find terrain height at current x
        # Find the two closest terrain points
        terrain_points = sorted(terrain_profile.keys())
        
        # Find the two terrain points that bracket the current position
        lower_point = None
        upper_point = None
        
        for point in terrain_points:
            if point <= current_x:
                lower_point = point
            if point >= current_x and upper_point is None:
                upper_point = point
        
        # Handle edge cases
        if lower_point is None:
            lower_point = terrain_points[0]
        if upper_point is None:
            upper_point = terrain_points[-1]
        
        # Interpolate height
        if lower_point == upper_point:
            terrain_height = terrain_profile[lower_point]
        else:
            # Linear interpolation
            fraction = (current_x - lower_point) / (upper_point - lower_point)
            terrain_height = terrain_profile[lower_point] + fraction * (terrain_profile[upper_point] - terrain_profile[lower_point])
        
        # Set aircraft height to terrain height + 150m (safe distance)
        position = (current_x, position[1], terrain_height + 150)
        
        # Adjust velocity to match terrain slope
        # This creates a more realistic flight path
        if i > 0:
            z_delta = position[2] - prev_position[2]
            z_velocity = z_delta / 0.5  # 0.5s time increment
            velocity = (velocity[0], velocity[1], z_velocity)
        
        # Remember previous position for velocity calculation
        prev_position = position
        
        # Send updated PDU
        aircraft_pdu = generator.create_moving_aircraft_pdu(
            entity_id=entity_id,
            position=position,
            velocity=velocity
        )
        generator.send_pdu(aircraft_pdu)
        
        # Report progress on some iterations
        if i % 5 == 0:
            print(f"Update {i+1}/30: Aircraft at position {position}, terrain height: {terrain_height}")
    
    print("Terrain following scenario completed")

def run_electronic_warfare_scenario(generator):
    """
    Scenario 7: Electronic Warfare
    Simulates radar, jamming, and electronic warfare activities
    
    Note: This is a simplified simulation as DIS doesn't directly support
    all aspects of EW without custom Signal PDUs
    """
    print("\n=== ELECTRONIC WARFARE SCENARIO ===")
    ew_logs = []
    # Create radar station (ground-based)
    print("Step 1: Creating radar station")
    radar_id = 60
    radar_position = (0.0, 0.0, 50.0)  # On a hill
    
    radar_pdu = generator.create_static_tank_pdu(  # Using tank PDU as a stand-in for radar
        entity_id=radar_id,
        position=radar_position
    )
    generator.send_pdu(radar_pdu)
    print(f"Created radar station (ID: {radar_id}) at position {radar_position}")
    
    # Create target aircraft
    print("\nStep 2: Creating target aircraft")
    target_id = 61
    target_position = (2000.0, 0.0, 1000.0)
    target_velocity = (-50.0, 0.0, 0.0)  # Moving west, toward radar
    
    target_pdu = generator.create_moving_aircraft_pdu(
        entity_id=target_id,
        position=target_position,
        velocity=target_velocity
    )
    generator.send_pdu(target_pdu)
    print(f"Created target aircraft (ID: {target_id}) at position {target_position}")
    
    # Create jamming aircraft
    print("\nStep 3: Creating jamming aircraft")
    jammer_id = 62
    jammer_position = (1800.0, 500.0, 1500.0)
    jammer_velocity = (-40.0, 0.0, 0.0)  # Moving west, accompanying target
    
    jammer_pdu = generator.create_moving_aircraft_pdu(
        entity_id=jammer_id,
        position=jammer_position,
        velocity=jammer_velocity
    )
    generator.send_pdu(jammer_pdu)
    print(f"Created jamming aircraft (ID: {jammer_id}) at position {jammer_position}")
    
    # Simulate electronic warfare scenario
    print("\nStep 4: Running EW scenario")
    
    # Define radar detection range
    radar_range = 1500.0
    jamming_active = False
    
    for i in range(20):
        # Wait a bit
        time.sleep(0.5)
        
        # Update positions
        target_position = (
            target_position[0] + target_velocity[0] * 0.5,
            target_position[1] + target_velocity[1] * 0.5,
            target_position[2] + target_velocity[2] * 0.5
        )
        
        jammer_position = (
            jammer_position[0] + jammer_velocity[0] * 0.5,
            jammer_position[1] + jammer_velocity[1] * 0.5,
            jammer_position[2] + jammer_velocity[2] * 0.5
        )
        
        # Send updated PDUs
        target_pdu = generator.create_moving_aircraft_pdu(
            entity_id=target_id,
            position=target_position,
            velocity=target_velocity
        )
        generator.send_pdu(target_pdu)
        
        jammer_pdu = generator.create_moving_aircraft_pdu(
            entity_id=jammer_id,
            position=jammer_position,
            velocity=jammer_velocity
        )
        generator.send_pdu(jammer_pdu)
        
        # Calculate distance to radar
        distance_to_radar = math.sqrt(
            (target_position[0] - radar_position[0])**2 +
            (target_position[1] - radar_position[1])**2 +
            (target_position[2] - radar_position[2])**2
        )
        
        # Determine if target is in radar range
        target_detected = distance_to_radar < radar_range
        
        # Activate jamming when in range
        if distance_to_radar < radar_range * 1.2 and not jamming_active:
            jamming_active = True
            print(f"\nUpdate {i+1}: Jamming aircraft activating countermeasures!")

            # Simulate jamming signal (digital noise)
            jamming_data = bytes([random.randint(0, 255) for _ in range(64)])
            signal_pdu = generator.create_signal_pdu(
                entity_id=jammer_id,
                radio_id=1,
                data_bytes=jamming_data,
                encoding_scheme=3  # Digital Data
            )
            generator.send_pdu(signal_pdu)
            print(f"Sent jamming SignalPDU from entity {jammer_id}")

        
        # Report
        if i % 4 == 0:
            status = "NOT DETECTED"
            
            if target_detected and not jamming_active:
                status = "DETECTED"
            elif target_detected and jamming_active:
                # Random chance to simulate radar blindness (e.g. 80% effective jamming)
                if random.random() < 0.8:
                    status = "JAMMED"
                else:
                    status = "DETECTED"

            print(f"Update {i+1}: Target at {target_position}, distance to radar: {distance_to_radar:.1f}m")
            print(f"  Radar status: {status}, Jamming: {'ACTIVE' if jamming_active else 'INACTIVE'}")

            if status == "JAMMED":
                print("  Radar cannot track target — signal disrupted!")

            # Optional: log status for analysis
            ew_logs.append({
                "tick": i,
                "target_position": target_position,
                "jammer_position": jammer_position,
                "distance_to_radar": distance_to_radar,
                "jamming_active": jamming_active,
                "radar_status": status
            })
            # pd.DataFrame(ew_logs).to_parquet("ew_scenario_radar_blindness.parquet")
            # print("Scenario logs saved to ew_scenario_radar_blindness.parquet")

    
    print("Electronic warfare scenario completed")

# ============================================================================
# Main Function (to be integrated with PDUGenerator)
# ============================================================================

def main():
    print("OpenDIS Scenario Scripts")
    print("====================")
    print("Using open-dis-python library to generate DIS PDUs")
    print()
    
    # Create PDU generator
    generator = PDUGenerator()
    
    while True:
        print("\nAvailable scenarios:")
        print("1. Static Vehicles")
        print("2. Moving Aircraft")
        print("3. Air Combat")
        print("4. Collision")
        print("5. Formation Flight")
        print("6. Terrain Following")
        print("7. Electronic Warfare")
        print("8. All Scenarios")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect scenario (0-7): "))
            
            if choice == 0:
                break
            elif choice == 1:
                run_static_vehicles_scenario(generator)
            elif choice == 2:
                run_moving_aircraft_scenario(generator)
            elif choice == 3:
                run_air_combat_scenario(generator)
            elif choice == 4:
                run_collision_scenario(generator)
            elif choice == 5:
                run_formation_flight_scenario(generator)
            elif choice == 6:
                run_terrain_following_scenario(generator)
            elif choice == 7:
                run_electronic_warfare_scenario(generator)
            elif choice == 8:
                run_static_vehicles_scenario(generator)
                run_moving_aircraft_scenario(generator)
                run_air_combat_scenario(generator)
                run_collision_scenario(generator)
                run_formation_flight_scenario(generator)
                run_terrain_following_scenario(generator)
                run_electronic_warfare_scenario(generator)
            else:
                print("Invalid choice. Please select 0-7.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running scenario: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nScenario complete.")
    
    # Clean up
    generator.close_socket()
    print("Scenarios closed. Exiting.")

if __name__ == "__main__":
    main()