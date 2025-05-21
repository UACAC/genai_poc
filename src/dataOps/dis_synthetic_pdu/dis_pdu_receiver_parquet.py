#!/usr/bin/env python3

import sys
import socket
import time
import datetime
import binascii
import traceback
import argparse
import os
import json
import pickle
from io import BytesIO
from pathlib import Path
import pandas as pd

# Import OpenDIS classes
from opendis.dis7 import *
from opendis.PduFactory import createPdu
from opendis.DataInputStream import DataInputStream

# Custom JSON encoder for DIS objects
class DISEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles DIS objects"""
    def default(self, obj):
        # Handle EventIdentifier
        if isinstance(obj, EventIdentifier):
            return {
                "siteID": obj.siteID,
                "applicationID": obj.applicationID,
                "eventNumber": obj.eventNumber
            }
        # Handle EntityID
        elif isinstance(obj, EntityID):
            return {
                "siteID": obj.siteID,
                "applicationID": obj.applicationID,
                "entityID": obj.entityID
            }
        # Handle EntityType
        elif isinstance(obj, EntityType):
            return {
                "entityKind": obj.entityKind,
                "domain": obj.domain,
                "country": obj.country,
                "category": obj.category,
                "subcategory": obj.subcategory,
                "specific": obj.specific,
                "extra": obj.extra
            }
        # Handle Vector3Double
        elif isinstance(obj, Vector3Double):
            return {
                "x": obj.x,
                "y": obj.y,
                "z": obj.z
            }
        # Handle Vector3Float
        elif isinstance(obj, Vector3Float):
            return {
                "x": obj.x,
                "y": obj.y,
                "z": obj.z
            }
        # Handle EulerAngles
        elif isinstance(obj, EulerAngles):
            return {
                "psi": obj.psi,
                "theta": obj.theta,
                "phi": obj.phi
            }
        # Handle EntityMarking
        elif isinstance(obj, EntityMarking):
            return {
                "characters": obj.characters if hasattr(obj, 'characters') else str(obj)
            }
        # Handle DeadReckoningParameters
        elif isinstance(obj, DeadReckoningParameters):
            return {
                "deadReckoningAlgorithm": obj.deadReckoningAlgorithm,
                "otherParameters": obj.otherParameters
            }
        # Handle bytes
        elif isinstance(obj, bytes):
            return binascii.hexlify(obj).decode('utf-8')
        # Handle any other DIS object with a __dict__
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            # Let the base class handle it (or raise TypeError)
            return super().default(obj)

class PDUReceiver:
    """Class for receiving and logging DIS PDUs using the open-dis library"""
    
    def __init__(self, port=3000, log_dir="pdu_logs", console_output=True, 
                save_binary=True, raw_hex=True, format='all'):
        """Initialize the PDU receiver"""
        self.port = port
        self.socket = None
        
        # Convert to absolute path and ensure it exists
        self.log_dir = os.path.abspath(log_dir)
        self.console_output = console_output
        self.save_binary = save_binary
        self.raw_hex = raw_hex
        self.format = format  # 'all', 'parquet', 'pickle', 'text', or 'json'
        self.log_file = None
        self.json_file = None
        self.binary_dir = os.path.join(self.log_dir, "binary")
        self.pdu_count = 0
        
        # For batch processing
        self.pdu_list = []  # Store PDUs for batch processing
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create directories immediately
        self.ensure_directories_exist()
        
        print(f"Initialized PDU Receiver. Log directory: {self.log_dir}")
        if self.save_binary:
            print(f"Binary directory: {self.binary_dir}")
        
    def ensure_directories_exist(self):
        """Create log directories if they don't exist"""
        try:
            # Create log directory
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            print(f"Ensured log directory exists: {self.log_dir}")
            
            # Create binary directory
            if self.save_binary:
                Path(self.binary_dir).mkdir(parents=True, exist_ok=True)
                print(f"Ensured binary directory exists: {self.binary_dir}")
                
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            traceback.print_exc()
            return False
        
    def setup_socket(self):
        """Set up the UDP socket for receiving PDUs"""
        try:
            # Create a plain UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Allow reuse of addresses
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the port on all interfaces
            self.socket.bind(("0.0.0.0", self.port))
            
            print(f"Socket bound to port {self.port} on all interfaces")
            return True
            
        except socket.error as e:
            print(f"Error creating socket: {e}")
            traceback.print_exc()
            return False
    
    def setup_logging(self):
        """Set up the log files"""
        try:
            # Directories should already exist from initialization
            # If not, ensure they do
            self.ensure_directories_exist()
            
            # Create log files based on format
            if self.format in ['all', 'text']:
                log_filename = os.path.join(self.log_dir, f"pdu_log_{self.timestamp}.txt")
                self.log_file = open(log_filename, "w")
                print(f"Logging to text file: {log_filename}")
            
            if self.format in ['all', 'json']:
                json_filename = os.path.join(self.log_dir, f"pdu_log_{self.timestamp}.json")
                self.json_file = open(json_filename, "w")
                self.json_file.write("[\n")  # Start JSON array
                print(f"Logging to JSON file: {json_filename}")
            
            # Parquet and pickle files will be created when saving PDUs
            if self.format in ['all', 'parquet']:
                print(f"Will save to parquet file: {os.path.join(self.log_dir, f'pdu_log_{self.timestamp}.parquet')}")
            
            if self.format in ['all', 'pickle']:
                print(f"Will save to pickle file: {os.path.join(self.log_dir, f'pdu_log_{self.timestamp}.pkl')}")
            
            return True
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            traceback.print_exc()
            return False
    
    def save_to_parquet(self):
        """Save collected PDUs to a parquet file"""
        if not self.pdu_list:
            print("No PDUs to save to parquet file")
            return
        
        try:
            # Convert to DataFrame
            df_data = []
            for entry in self.pdu_list:
                # Convert any complex objects to strings
                entry_copy = entry.copy()
                for key, value in entry.items():
                    if isinstance(value, dict):
                        entry_copy[key] = json.dumps(value, cls=DISEncoder)
                df_data.append(entry_copy)
            
            df = pd.DataFrame(df_data)
            
            # Save to parquet
            parquet_path = os.path.join(self.log_dir, f"pdu_log_{self.timestamp}.parquet")
            df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            print(f"Saved {len(self.pdu_list)} PDUs to parquet file: {parquet_path}")
            
            # Verify the file was created
            if os.path.exists(parquet_path):
                file_size = os.path.getsize(parquet_path)
                print(f"  Parquet file size: {file_size} bytes")
            else:
                print(f"  WARNING: Parquet file not found after writing!")
                
        except Exception as e:
            print(f"Error saving to parquet: {e}")
            traceback.print_exc()
    
    def save_to_pickle(self):
        """Save collected PDUs to a pickle file"""
        if not self.pdu_list:
            print("No PDUs to save to pickle file")
            return
            
        try:
            pickle_path = os.path.join(self.log_dir, f"pdu_log_{self.timestamp}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.pdu_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(self.pdu_list)} PDUs to pickle file: {pickle_path}")
            
            # Verify the file was created
            if os.path.exists(pickle_path):
                file_size = os.path.getsize(pickle_path)
                print(f"  Pickle file size: {file_size} bytes")
            else:
                print(f"  WARNING: Pickle file not found after writing!")
                
        except Exception as e:
            print(f"Error saving to pickle: {e}")
            traceback.print_exc()
    
    def close(self):
        """Close the socket and log files"""
        if self.socket:
            self.socket.close()
            self.socket = None
        
        # Close text log file
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        
        # Close and finalize JSON file
        if self.json_file:
            self.json_file.write("\n]")  # End JSON array
            self.json_file.close()
            self.json_file = None
        
        # Save to parquet and pickle if needed
        if self.format in ['all', 'parquet']:
            self.save_to_parquet()
        
        if self.format in ['all', 'pickle']:
            self.save_to_pickle()
    
    def receive_and_log(self, timeout=None):
        """
        Receive and log PDUs. If timeout is specified, receive for that many seconds.
        If timeout is None, receive indefinitely.
        """
        if not self.socket and not self.setup_socket():
            print("Error: Socket not available")
            return False
        
        if not self.setup_logging():
            print("Error: Logging setup failed")
            return False
        
        # Set socket timeout if specified
        if timeout:
            self.socket.settimeout(timeout)
        
        start_time = time.time()
        print(f"Starting PDU receiver{f' (timeout: {timeout}s)' if timeout else ''}...")
        print("Press Ctrl+C to stop receiving")
        
        try:
            while True:
                # Check if timeout has been reached
                if timeout and (time.time() - start_time) > timeout:
                    print(f"Timeout of {timeout}s reached")
                    break
                
                try:
                    # Receive data
                    data, addr = self.socket.recvfrom(4096)
                    
                    # Get current time for timestamp
                    timestamp = datetime.datetime.now()
                    
                    # Process and log the PDU
                    self.process_pdu(data, addr, timestamp)
                    
                except socket.timeout:
                    continue
                except socket.error as e:
                    print(f"Socket error: {e}")
                    traceback.print_exc()
                    break
                
        except KeyboardInterrupt:
            print("\nReceiving stopped by user")
        
        print(f"Received and logged {self.pdu_count} PDUs")
        return True
    
    def process_pdu(self, data, addr, timestamp):
        """Process a received PDU and log it"""
        try:
            # Create BytesIO buffer for deserialization
            memoryStream = BytesIO(data)
            
            # Create DataInputStream with the buffer
            data_stream = DataInputStream(memoryStream)
            
            # Create PDU using PduFactory
            pdu = createPdu(data)
            
            # Increment counter
            self.pdu_count += 1
            
            # Log the PDU to console if enabled
            if self.console_output:
                print(f"\n[{timestamp}] Received PDU #{self.pdu_count} from {addr[0]}:{addr[1]} ({len(data)} bytes)")
                print(f"PDU Type: {pdu.__class__.__name__ if pdu else 'Unknown'}")
                
                # If it's a known PDU type that we can parse
                if pdu is not None:
                    # Print a few key details
                    print(f"  Protocol Version: {pdu.protocolVersion}")
                    print(f"  PDU Type: {pdu.pduType}")
                    print(f"  Exercise ID: {pdu.exerciseID}")
                    
                    # Print specific attributes based on PDU type
                    if isinstance(pdu, EntityStatePdu):
                        print(f"  Entity ID: {pdu.entityID.entityID}")
                        print(f"  Position: ({pdu.entityLocation.x}, {pdu.entityLocation.y}, {pdu.entityLocation.z})")
                        if hasattr(pdu, 'marking') and pdu.marking:
                            print(f"  Marking: {pdu.marking.characters}")
                    elif isinstance(pdu, FirePdu):
                        print(f"  Firing Entity ID: {pdu.firingEntityID.entityID}")
                        print(f"  Target Entity ID: {pdu.targetEntityID.entityID}")
                        print(f"  Munition Expendable ID: {pdu.munitionExpendableID.entityID if hasattr(pdu, 'munitionExpendableID') else 'N/A'}")
                    elif isinstance(pdu, DetonationPdu):
                        print(f"  Firing Entity ID: {pdu.firingEntityID.entityID}")
                        print(f"  Target Entity ID: {pdu.targetEntityID.entityID}")
                        print(f"  Detonation Result: {pdu.detonationResult}")
                    elif isinstance(pdu, CollisionPdu):
                        print(f"  Issuing Entity ID: {pdu.issuingEntityID.entityID}")
                        print(f"  Colliding Entity ID: {pdu.collidingEntityID.entityID}")
                    elif isinstance(pdu, SignalPdu):
                        print(f"  Entity ID: {pdu.entityID.entityID}")
                        print(f"  Radio ID: {pdu.radioID}")
                        print(f"  Encoding Scheme: {pdu.encodingScheme}")
                
                # Print raw hex if enabled
                if self.raw_hex:
                    print(f"  Raw Data (hex): {binascii.hexlify(data).decode('utf-8')}")
            
            # Save the binary PDU if enabled
            if self.save_binary:
                # Double-check that binary directory exists
                if not os.path.exists(self.binary_dir):
                    os.makedirs(self.binary_dir)
                    
                binary_filename = os.path.join(
                    self.binary_dir, 
                    f"pdu_{self.pdu_count}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.bin"
                )
                
                try:
                    with open(binary_filename, "wb") as binary_file:
                        binary_file.write(data)
                except Exception as e:
                    print(f"Error saving binary file: {e}")
                    print(f"Attempted to save to: {binary_filename}")
            
            # Prepare log entry with manual attribute extraction to avoid JSON serialization issues
            log_entry = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "pdu_number": self.pdu_count,
                "source_ip": addr[0],
                "source_port": addr[1],
                "size_bytes": len(data),
                "raw_data_hex": binascii.hexlify(data).decode('utf-8')
            }
            
            if pdu is not None:
                # Add PDU type and common attributes
                log_entry["pdu_type"] = pdu.__class__.__name__
                log_entry["protocol_version"] = pdu.protocolVersion
                log_entry["pdu_type_value"] = pdu.pduType
                log_entry["exercise_id"] = pdu.exerciseID
                
                # Add specific attributes based on PDU type
                if isinstance(pdu, EntityStatePdu):
                    log_entry["entity_id"] = {
                        "site": pdu.entityID.siteID,
                        "application": pdu.entityID.applicationID,
                        "entity": pdu.entityID.entityID
                    }
                    log_entry["force_id"] = pdu.forceId
                    log_entry["entity_type"] = {
                        "kind": pdu.entityType.entityKind,
                        "domain": pdu.entityType.domain,
                        "country": pdu.entityType.country,
                        "category": pdu.entityType.category,
                        "subcategory": pdu.entityType.subcategory,
                        "specific": pdu.entityType.specific,
                        "extra": pdu.entityType.extra
                    }
                    log_entry["location"] = {
                        "x": pdu.entityLocation.x,
                        "y": pdu.entityLocation.y,
                        "z": pdu.entityLocation.z
                    }
                    log_entry["orientation"] = {
                        "psi": pdu.entityOrientation.psi,
                        "theta": pdu.entityOrientation.theta,
                        "phi": pdu.entityOrientation.phi
                    }
                    log_entry["velocity"] = {
                        "x": pdu.entityLinearVelocity.x,
                        "y": pdu.entityLinearVelocity.y,
                        "z": pdu.entityLinearVelocity.z
                    }
                    if hasattr(pdu, 'marking') and pdu.marking:
                        log_entry["marking"] = pdu.marking.characters
                
                elif isinstance(pdu, FirePdu):
                    # Extract firing entity ID
                    if hasattr(pdu, 'firingEntityID'):
                        log_entry["firing_entity_id"] = {
                            "site": pdu.firingEntityID.siteID,
                            "application": pdu.firingEntityID.applicationID,
                            "entity": pdu.firingEntityID.entityID
                        }
                    
                    # Extract target entity ID
                    if hasattr(pdu, 'targetEntityID'):
                        log_entry["target_entity_id"] = {
                            "site": pdu.targetEntityID.siteID,
                            "application": pdu.targetEntityID.applicationID,
                            "entity": pdu.targetEntityID.entityID
                        }
                    
                    # Handle either munitionID or munitionExpendableID
                    if hasattr(pdu, 'munitionExpendableID'):
                        log_entry["munition_expendable_id"] = {
                            "site": pdu.munitionExpendableID.siteID,
                            "application": pdu.munitionExpendableID.applicationID,
                            "entity": pdu.munitionExpendableID.entityID
                        }
                    elif hasattr(pdu, 'munitionID'):
                        log_entry["munition_id"] = {
                            "site": pdu.munitionID.siteID,
                            "application": pdu.munitionID.applicationID,
                            "entity": pdu.munitionID.entityID
                        }
                    
                    # Handle event ID if it exists
                    if hasattr(pdu, 'eventID'):
                        if isinstance(pdu.eventID, EventIdentifier):
                            log_entry["event_id"] = {
                                "site": pdu.eventID.siteID,
                                "application": pdu.eventID.applicationID,
                                "event": pdu.eventID.eventNumber
                            }
                        else:
                            # Handle if eventID is not the expected type
                            log_entry["event_id"] = str(pdu.eventID)
                    
                    # Handle location fields with different possible names
                    if hasattr(pdu, 'locationInWorldCoordinates'):
                        log_entry["location"] = {
                            "x": pdu.locationInWorldCoordinates.x,
                            "y": pdu.locationInWorldCoordinates.y,
                            "z": pdu.locationInWorldCoordinates.z
                        }
                    elif hasattr(pdu, 'location'):
                        log_entry["location"] = {
                            "x": pdu.location.x,
                            "y": pdu.location.y,
                            "z": pdu.location.z
                        }
                    
                    # Handle velocity fields with different possible names
                    if hasattr(pdu, 'initialVelocity'):
                        log_entry["velocity"] = {
                            "x": pdu.initialVelocity.x,
                            "y": pdu.initialVelocity.y,
                            "z": pdu.initialVelocity.z
                        }
                    elif hasattr(pdu, 'velocity'):
                        log_entry["velocity"] = {
                            "x": pdu.velocity.x,
                            "y": pdu.velocity.y,
                            "z": pdu.velocity.z
                        }
                    
                    # Add other fields if they exist
                    if hasattr(pdu, 'warhead'):
                        log_entry["warhead"] = pdu.warhead
                    if hasattr(pdu, 'fuse'):
                        log_entry["fuse"] = pdu.fuse
                    if hasattr(pdu, 'quantity'):
                        log_entry["quantity"] = pdu.quantity
                    if hasattr(pdu, 'rate'):
                        log_entry["rate"] = pdu.rate
                
                elif isinstance(pdu, DetonationPdu):
                    # Extract firing entity ID
                    if hasattr(pdu, 'firingEntityID'):
                        log_entry["firing_entity_id"] = {
                            "site": pdu.firingEntityID.siteID,
                            "application": pdu.firingEntityID.applicationID,
                            "entity": pdu.firingEntityID.entityID
                        }
                    
                    # Extract target entity ID
                    if hasattr(pdu, 'targetEntityID'):
                        log_entry["target_entity_id"] = {
                            "site": pdu.targetEntityID.siteID,
                            "application": pdu.targetEntityID.applicationID,
                            "entity": pdu.targetEntityID.entityID
                        }
                    
                    # Handle either munitionID or munitionExpendableID
                    if hasattr(pdu, 'munitionExpendableID'):
                        log_entry["munition_expendable_id"] = {
                            "site": pdu.munitionExpendableID.siteID,
                            "application": pdu.munitionExpendableID.applicationID,
                            "entity": pdu.munitionExpendableID.entityID
                        }
                    elif hasattr(pdu, 'munitionID'):
                        log_entry["munition_id"] = {
                            "site": pdu.munitionID.siteID,
                            "application": pdu.munitionID.applicationID,
                            "entity": pdu.munitionID.entityID
                        }
                    
                    # Handle event ID
                    if hasattr(pdu, 'eventID'):
                        if isinstance(pdu.eventID, EventIdentifier):
                            log_entry["event_id"] = {
                                "site": pdu.eventID.siteID,
                                "application": pdu.eventID.applicationID,
                                "event": pdu.eventID.eventNumber
                            }
                        else:
                            # Just use whatever is there
                            log_entry["event_id"] = str(pdu.eventID)
                    
                    # Handle location with different possible field names
                    if hasattr(pdu, 'locationInWorldCoordinates'):
                        log_entry["location"] = {
                            "x": pdu.locationInWorldCoordinates.x,
                            "y": pdu.locationInWorldCoordinates.y,
                            "z": pdu.locationInWorldCoordinates.z
                        }
                    elif hasattr(pdu, 'location'):
                        log_entry["location"] = {
                            "x": pdu.location.x,
                            "y": pdu.location.y,
                            "z": pdu.location.z
                        }
                    
                    # Handle velocity
                    if hasattr(pdu, 'velocity'):
                        log_entry["velocity"] = {
                            "x": pdu.velocity.x,
                            "y": pdu.velocity.y,
                            "z": pdu.velocity.z
                        }
                    
                    # Add detonation result
                    if hasattr(pdu, 'detonationResult'):
                        log_entry["detonation_result"] = pdu.detonationResult
                
                elif isinstance(pdu, CollisionPdu):
                    # Extract issuing entity ID
                    if hasattr(pdu, 'issuingEntityID'):
                        log_entry["issuing_entity_id"] = {
                            "site": pdu.issuingEntityID.siteID,
                            "application": pdu.issuingEntityID.applicationID,
                            "entity": pdu.issuingEntityID.entityID
                        }
                    
                    # Extract colliding entity ID
                    if hasattr(pdu, 'collidingEntityID'):
                        log_entry["colliding_entity_id"] = {
                            "site": pdu.collidingEntityID.siteID,
                            "application": pdu.collidingEntityID.applicationID,
                            "entity": pdu.collidingEntityID.entityID
                        }
                    
                    # Handle event ID field
                    if hasattr(pdu, 'eventID'):
                        if isinstance(pdu.eventID, EventIdentifier):
                            log_entry["event_id"] = {
                                "site": pdu.eventID.siteID,
                                "application": pdu.eventID.applicationID,
                                "event": pdu.eventID.eventNumber
                            }
                        else:
                            # Just use whatever is there
                            log_entry["event_id"] = str(pdu.eventID)
                    
                    # Handle collision type
                    if hasattr(pdu, 'collisionType'):
                        log_entry["collision_type"] = pdu.collisionType
                    
                    # Handle velocity
                    if hasattr(pdu, 'velocity'):
                        log_entry["velocity"] = {
                            "x": pdu.velocity.x,
                            "y": pdu.velocity.y,
                            "z": pdu.velocity.z
                        }
                    
                    # Handle mass
                    if hasattr(pdu, 'mass'):
                        log_entry["mass"] = pdu.mass
                    
                    # Handle location with different possible field names
                    if hasattr(pdu, 'locationInWorldCoordinates'):
                        log_entry["location"] = {
                            "x": pdu.locationInWorldCoordinates.x,
                            "y": pdu.locationInWorldCoordinates.y,
                            "z": pdu.locationInWorldCoordinates.z
                        }
                    elif hasattr(pdu, 'location'):
                        log_entry["location"] = {
                            "x": pdu.location.x,
                            "y": pdu.location.y,
                            "z": pdu.location.z
                        }
                
                elif isinstance(pdu, SignalPdu):
                    if hasattr(pdu, 'entityID'):
                        log_entry["entity_id"] = {
                            "site": pdu.entityID.siteID,
                            "application": pdu.entityID.applicationID,
                            "entity": pdu.entityID.entityID
                        }
                    
                    if hasattr(pdu, 'radioID'):
                        log_entry["radio_id"] = pdu.radioID
                    
                    if hasattr(pdu, 'encodingScheme'):
                        log_entry["encoding_scheme"] = pdu.encodingScheme
                    
                    if hasattr(pdu, 'tdlType'):
                        log_entry["tdl_type"] = pdu.tdlType
                    
                    if hasattr(pdu, 'sampleRate'):
                        log_entry["sample_rate"] = pdu.sampleRate
                    
                    if hasattr(pdu, 'samples'):
                        log_entry["samples"] = pdu.samples
                    
                    if hasattr(pdu, 'data') and pdu.data:
                        log_entry["data_length"] = len(pdu.data)
            
            # Add the log entry to our collection for batch save
            self.pdu_list.append(log_entry)
            
            # Write to text log file
            if self.log_file:
                try:
                    self.log_file.write(json.dumps(log_entry, cls=DISEncoder) + "\n")
                    self.log_file.flush()
                except Exception as e:
                    print(f"Error writing to text log: {e}")
                    traceback.print_exc()
            
            # Write to JSON file
            if self.json_file:
                try:
                    # Add comma except for first item
                    if self.pdu_count > 1:
                        self.json_file.write(",\n")
                    self.json_file.write(json.dumps(log_entry, cls=DISEncoder, indent=2))
                    self.json_file.flush()
                except Exception as e:
                    print(f"Error writing to JSON file: {e}")
                    traceback.print_exc()
            
        except Exception as e:
            print(f"Error processing PDU: {e}")
            traceback.print_exc()