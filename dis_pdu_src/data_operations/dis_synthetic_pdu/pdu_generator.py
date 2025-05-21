#!/usr/bin/env python3

import sys
import time
import socket
import math
import traceback
from io import BytesIO

# Import OpenDIS classes
from opendis.dis7 import *
from opendis.DataOutputStream import DataOutputStream

# Constants for simulation
DEFAULT_PORT = 3000
DEFAULT_MULTICAST_GROUP = "239.1.2.3"
LOCALHOST = "127.0.0.1"
DEFAULT_EXERCISE_ID = 1

class PDUGenerator:
    """Class for generating and sending DIS PDUs using the open-dis library"""
    
    def __init__(self, port=DEFAULT_PORT, multicast_group=DEFAULT_MULTICAST_GROUP, exercise_id=DEFAULT_EXERCISE_ID):
        """Initialize the PDU generator"""
        self.port = port
        self.multicast_group = multicast_group
        self.localhost = LOCALHOST
        self.exercise_id = exercise_id
        self.socket = None
        self.entity_counter = 1
        self.event_counter = 1
        
    def setup_socket(self):
        """Set up the UDP socket for sending PDUs"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
            
            return True
        except socket.error as e:
            print(f"Error creating socket: {e}")
            return False
    
    def close_socket(self):
        """Close the socket"""
        if self.socket:
            self.socket.close()
            self.socket = None
    
    def send_pdu(self, pdu):
        """Serialize and send a PDU"""
        if not self.socket and not self.setup_socket():
            print("Error: Socket not available")
            return False
        
        try:
            # Set common PDU fields
            pdu.protocolVersion = 7
            pdu.exerciseID = self.exercise_id
            
            # Ensure all required fields are initialized
            if not hasattr(pdu, 'pduStatus') or pdu.pduStatus is None:
                pdu.pduStatus = 0
                
            if not hasattr(pdu, 'padding') or pdu.padding is None:
                pdu.padding = 0
            
            # Create BytesIO buffer for serialization
            memoryStream = BytesIO()
            
            # Create DataOutputStream with the buffer
            data_stream = DataOutputStream(memoryStream)
            
            # Serialize the PDU to the stream
            pdu.serialize(data_stream)
            
            # Get the bytes from the BytesIO buffer
            pdu_bytes = memoryStream.getvalue()
            
            # Send to multicast group
            self.socket.sendto(pdu_bytes, (self.localhost, self.port))
            
            # Print info about the sent PDU
            print(f"Sent {pdu.__class__.__name__} ({len(pdu_bytes)} bytes) to {self.localhost}:{self.port}")
            return True
        except Exception as e:
            print(f"Error sending PDU: {e}")
            traceback.print_exc()
            return False
    
    def next_entity_id(self):
        """Get the next entity ID"""
        entity_id = self.entity_counter
        self.entity_counter += 1
        return entity_id
    
    def next_event_id(self):
        """Get the next event ID"""
        event_id = self.event_counter
        self.event_counter += 1
        return event_id
    
    def create_entity_id(self, entity_id=None):
        """Create an EntityID object"""
        eid = EntityID()
        eid.siteID = 1
        eid.applicationID = 1
        eid.entityID = entity_id if entity_id is not None else self.next_entity_id()
        return eid
    
    def create_event_id(self):
        """Create an EventID object"""
        # Using EventIdentifier from your code
        eid = EventIdentifier()
        eid.siteID = 1
        eid.applicationID = 1
        eid.eventNumber = self.next_event_id()
        return eid
    
    def create_entity_type(self, kind, domain, country, category, subcategory=0, specific=0, extra=0):
        """Create an EntityType object"""
        entity_type = EntityType()
        entity_type.entityKind = kind
        entity_type.domain = domain
        entity_type.country = country
        entity_type.category = category
        entity_type.subcategory = subcategory
        entity_type.specific = specific
        entity_type.extra = extra
        return entity_type
    
    def create_vector3_double(self, x, y, z):
        """Create a Vector3Double object"""
        vector = Vector3Double()
        vector.x = x
        vector.y = y
        vector.z = z
        return vector
    
    def create_vector3_float(self, x, y, z):
        """Create a Vector3Float object"""
        vector = Vector3Float()
        vector.x = x
        vector.y = y
        vector.z = z
        return vector
    
    def create_euler_angles(self, psi, theta, phi):
        """Create an EulerAngles object (radians)"""
        orientation = EulerAngles()
        orientation.psi = psi
        orientation.theta = theta
        orientation.phi = phi
        return orientation
    
    def create_static_tank_pdu(self, entity_id=None, position=(0.0, 0.0, 0.0)):
        """Create an Entity State PDU for a stationary tank"""
        espdu = EntityStatePdu()
        
        # Initialize required fields
        espdu.protocolVersion = 7
        espdu.exerciseID = self.exercise_id
        espdu.pduType = 1  # Entity State PDU
        espdu.protocolFamily = 1  # Entity Information
        espdu.timestamp = 0
        espdu.length = 144  # Minimum size
        espdu.pduStatus = 0
        espdu.padding = 0
        
        # Entity ID
        espdu.entityID = self.create_entity_id(entity_id)
        
        # Force ID
        espdu.forceId = 1  # Friendly
        
        # Entity Type (Platform, Land, USA, Tank)
        espdu.entityType = self.create_entity_type(1, 1, 225, 1)
        
        # Alternative Entity Type
        espdu.alternativeEntityType = self.create_entity_type(0, 0, 0, 0)
        
        # Location and orientation
        espdu.entityLocation = self.create_vector3_double(*position)
        espdu.entityOrientation = self.create_euler_angles(0.0, 0.0, 0.0)
        
        # Linear velocity (stationary)
        espdu.entityLinearVelocity = self.create_vector3_float(0.0, 0.0, 0.0)
        
        # Dead reckoning parameters
        espdu.deadReckoningParameters = DeadReckoningParameters()
        espdu.deadReckoningParameters.deadReckoningAlgorithm = 0  # None
        espdu.deadReckoningParameters.otherParameters = [0] * 15
        
        # Entity appearance
        espdu.entityAppearance = 0
        
        # Entity marking (using EntityMarking as in your code)
        espdu.marking = EntityMarking()
        espdu.marking.setString("TANK0001")
        
        # Entity capabilities
        espdu.capabilities = 0
        
        return espdu
    
    def create_moving_aircraft_pdu(self, entity_id=None, position=(1000.0, 1000.0, 5000.0), velocity=(100.0, 0.0, 0.0)):
        """Create an Entity State PDU for an aircraft in motion"""
        espdu = EntityStatePdu()
        
        # Initialize required fields
        espdu.protocolVersion = 7
        espdu.exerciseID = self.exercise_id
        espdu.pduType = 1  # Entity State PDU
        espdu.protocolFamily = 1  # Entity Information
        espdu.timestamp = 0
        espdu.length = 144  # Minimum size
        espdu.pduStatus = 0
        espdu.padding = 0
        
        # Entity ID
        espdu.entityID = self.create_entity_id(entity_id)
        
        # Force ID
        espdu.forceId = 1  # Friendly
        
        # Entity Type (Platform, Air, USA, Fighter)
        espdu.entityType = self.create_entity_type(1, 2, 225, 2)
        
        # Alternative Entity Type
        espdu.alternativeEntityType = self.create_entity_type(0, 0, 0, 0)
        
        # Location and orientation
        espdu.entityLocation = self.create_vector3_double(*position)
        espdu.entityOrientation = self.create_euler_angles(0.0, 0.0, 0.0)
        
        # Linear velocity
        espdu.entityLinearVelocity = self.create_vector3_float(*velocity)
        
        # Dead reckoning parameters
        espdu.deadReckoningParameters = DeadReckoningParameters()
        espdu.deadReckoningParameters.deadReckoningAlgorithm = 2  # First order linear
        espdu.deadReckoningParameters.otherParameters = [0] * 15
        
        # Entity appearance
        espdu.entityAppearance = 0
        
        # Entity marking (using EntityMarking as in your code)
        espdu.marking = EntityMarking()
        espdu.marking.setString("EAGLE01")
        
        # Entity capabilities
        espdu.capabilities = 0
        
        return espdu
    
    def create_fire_pdu(self, firing_entity_id, target_entity_id, munition_entity_id=None,
                        fire_position=(1000.0, 1000.0, 5000.0), velocity=(500.0, 0.0, 0.0)):
        """Create a Fire PDU"""
        fire_pdu = FirePdu()
        
        # Initialize required fields
        fire_pdu.protocolVersion = 7
        fire_pdu.exerciseID = self.exercise_id
        fire_pdu.pduType = 2  # Fire PDU
        fire_pdu.protocolFamily = 2  # Warfare
        fire_pdu.timestamp = 0
        fire_pdu.length = 64  # Fire PDU size
        fire_pdu.pduStatus = 0
        fire_pdu.padding = 0
        
        # Entity IDs
        fire_pdu.firingEntityID = self.create_entity_id(firing_entity_id)
        fire_pdu.targetEntityID = self.create_entity_id(target_entity_id)
        
        # Munition ID (new entity)
        if munition_entity_id is None:
            munition_entity_id = self.next_entity_id()
        fire_pdu.munitionID = self.create_entity_id(munition_entity_id)
        
        # Event ID
        fire_pdu.eventID = self.create_event_id()
        
        # Fire location and velocity
        fire_pdu.locationInWorldCoordinates = self.create_vector3_double(*fire_position)
        fire_pdu.initialVelocity = self.create_vector3_float(*velocity)
        
        # Munition type (Munition, Air, USA, Missile)
        fire_pdu.munitionType = self.create_entity_type(2, 2, 225, 2)
        
        # Warhead, fuse, quantity, rate
        fire_pdu.warhead = 1000  # High explosive
        fire_pdu.fuse = 1        # Proximity
        fire_pdu.quantity = 1
        fire_pdu.rate = 0
        
        return fire_pdu
    
    def create_detonation_pdu(self, firing_entity_id, target_entity_id, munition_entity_id,
                             detonation_position=(1500.0, 1000.0, 5000.0), velocity=(450.0, 0.0, 0.0),
                             detonation_result=1):  # 1 = Entity Impact
        """Create a Detonation PDU"""
        detonation_pdu = DetonationPdu()
        
        # Initialize required fields
        detonation_pdu.protocolVersion = 7
        detonation_pdu.exerciseID = self.exercise_id
        detonation_pdu.pduType = 3  # Detonation PDU
        detonation_pdu.protocolFamily = 2  # Warfare
        detonation_pdu.timestamp = 0
        detonation_pdu.length = 88  # Detonation PDU size
        detonation_pdu.pduStatus = 0
        detonation_pdu.padding = 0
        
        # Entity IDs
        detonation_pdu.firingEntityID = self.create_entity_id(firing_entity_id)
        detonation_pdu.targetEntityID = self.create_entity_id(target_entity_id)
        detonation_pdu.munitionID = self.create_entity_id(munition_entity_id)
        
        # Event ID (should match the fire event for related detonations)
        detonation_pdu.eventID = self.create_event_id()
        
        # Location and velocity
        detonation_pdu.locationInWorldCoordinates = self.create_vector3_double(*detonation_position)
        detonation_pdu.velocity = self.create_vector3_float(*velocity)
        
        # Munition type (Munition, Air, USA, Missile)
        detonation_pdu.munitionType = self.create_entity_type(2, 2, 225, 2)
        
        # Warhead, fuse, quantity, rate
        detonation_pdu.warhead = 1000  # High explosive
        detonation_pdu.fuse = 1        # Proximity
        detonation_pdu.quantity = 1
        detonation_pdu.rate = 0
        
        # Detonation result
        detonation_pdu.detonationResult = detonation_result
        
        # Articulation parameters
        detonation_pdu.numberOfArticulationParameters = 0
        detonation_pdu.articulationParameters = []
        
        return detonation_pdu
    
    def create_collision_pdu(self, entity_id, collided_entity_id, 
                            collision_position=(1000.0, 1000.0, 0.0), velocity=(10.0, 0.0, 0.0), 
                            mass=20000.0, collision_type=1):
        """Create a Collision PDU"""
        collision_pdu = CollisionPdu()
        
        # Initialize required fields
        collision_pdu.protocolVersion = 7
        collision_pdu.exerciseID = self.exercise_id
        collision_pdu.pduType = 4  # Collision PDU
        collision_pdu.protocolFamily = 4  # Collision Family
        collision_pdu.timestamp = 0
        collision_pdu.length = 56  # Collision PDU size
        collision_pdu.pduStatus = 0
        collision_pdu.padding = 0
        
        # Entity IDs
        collision_pdu.issuingEntityID = self.create_entity_id(entity_id)
        collision_pdu.collidingEntityID = self.create_entity_id(collided_entity_id)
        
        # Event ID
        collision_pdu.eventID = self.create_event_id()
        
        # Collision type
        collision_pdu.collisionType = collision_type
        if hasattr(collision_pdu, 'pad'):
            collision_pdu.pad = 0  # Padding field
        
        # Velocity
        collision_pdu.velocity = self.create_vector3_float(*velocity)
        
        # Mass
        collision_pdu.mass = mass
        
        # Location
        collision_pdu.locationInWorldCoordinates = self.create_vector3_double(*collision_position)
        
        return collision_pdu
    
    def create_signal_pdu(self, entity_id, radio_id, data_bytes, encoding_scheme=1, sample_rate=44100, samples=1):
        """
        Create a Signal PDU for radio communication or jamming.
        encoding_scheme: 1 = Other, 2 = Voice, 3 = Digital Data, 4 = PCM
        """
        signal_pdu = SignalPdu()

        signal_pdu.protocolVersion = 7
        signal_pdu.exerciseID = self.exercise_id
        signal_pdu.pduType = 26  # Signal PDU
        signal_pdu.protocolFamily = 4
        signal_pdu.timestamp = 0
        signal_pdu.pduStatus = 0
        signal_pdu.padding = 0

        # Radio identifiers
        signal_pdu.entityID = self.create_entity_id(entity_id)
        signal_pdu.radioID = radio_id

        # Signal parameters
        signal_pdu.encodingScheme = encoding_scheme
        signal_pdu.tdlType = 0
        signal_pdu.sampleRate = sample_rate
        signal_pdu.samples = samples
        signal_pdu.data = data_bytes

        return signal_pdu


# ============================================================================
# Scenario Functions - Matching Scenario Script
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
                collision_type=1  # Elastic collision
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