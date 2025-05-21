#!/usr/bin/env python3

import sys
import socket
import struct
import time
import datetime
import binascii
import traceback
import argparse
import os
import json
from io import BytesIO

# Import OpenDIS classes
from opendis.dis7 import *
from opendis.PduFactory import createPdu
from opendis.DataInputStream import DataInputStream
import pandas as pd

class PDUReceiver:
    """Class for receiving and logging DIS PDUs using the open-dis library"""
    
    def __init__(self, port=3000, multicast_group="127.0.0.1", log_dir="pdu_logs", 
                console_output=True, save_binary=True, raw_hex=True):
        """Initialize the PDU receiver"""
        self.port = port
        self.multicast_group = multicast_group
        self.localhost = '127.0.0.1'
        self.socket = None
        self.log_dir = log_dir
        self.console_output = console_output
        self.save_binary = save_binary
        self.raw_hex = raw_hex
        self.log_file = None
        self.binary_dir = os.path.join(log_dir, "binary")
        self.pdu_count = 0
        
    def setup_socket(self):
        """Set up the UDP socket for receiving PDUs"""
        try:
            # Create the socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            
            # Allow reuse of addresses
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the port
            self.socket.bind(("0.0.0.0", self.port))
            
            print(f"Socket bound to port {self.port} and joined multicast group {self.multicast_group}")
            return True
            
        except socket.error as e:
            print(f"Error creating socket: {e}")
            traceback.print_exc()
            return False
    
    def setup_logging(self):
        """Set up the log directory and file"""
        try:
            # Create log directory if it doesn't exist
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            # Create binary directory if needed
            if self.save_binary and not os.path.exists(self.binary_dir):
                os.makedirs(self.binary_dir)
            
            # Create log file with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = os.path.join(self.log_dir, f"pdu_log_{timestamp}.txt")
            self.log_file = open(log_filename, "w")
            
            print(f"Logging to file: {log_filename}")
            
           
            
            return True
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            traceback.print_exc()
            return False
    
    def close(self):
        """Close the socket and log file"""
        if self.socket:
            self.socket.close()
            self.socket = None
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def receive_and_log(self, timeout=None):
        """
        Receive and log PDUs. If timeout is specified, receive for that many seconds.
        If timeout is None, receive indefinitely.
        """
        if not self.socket and not self.setup_socket():
            print("Error: Socket not available")
            return False
        
        if not self.log_file and not self.setup_logging():
            print("Error: Log file not available")
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
                print(f"PDU Type: {pdu.__class__.__name__}")
                
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
                        print(f"  Munition Expendable ID: {pdu.munitionExpendableID.entityID}")
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
                binary_filename = os.path.join(
                    self.binary_dir, 
                    f"pdu_{self.pdu_count}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.bin"
                )
                with open(binary_filename, "wb") as binary_file:
                    binary_file.write(data)
            
            # Log to file
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
                    log_entry["firing_entity_id"] = {
                        "site": pdu.firingEntityID.siteID,
                        "application": pdu.firingEntityID.applicationID,
                        "entity": pdu.firingEntityID.entityID
                    }
                    log_entry["target_entity_id"] = {
                        "site": pdu.targetEntityID.siteID,
                        "application": pdu.targetEntityID.applicationID,
                        "entity": pdu.targetEntityID.entityID
                    }
                    log_entry["munition_expendable_id"] = {
                        "site": pdu.munitionExpendableID.siteID,
                        "application": pdu.munitionExpendableID.applicationID,
                        "entity": pdu.munitionExpendableID.entityID
                    }
                    log_entry["location"] = {
                        "x": pdu.location.x,
                        "y": pdu.location.y,
                        "z": pdu.location.z
                    }
                    log_entry["velocity"] = {
                        "x": pdu.velocity.x,
                        "y": pdu.velocity.y,
                        "z": pdu.velocity.z
                    }
                
                elif isinstance(pdu, DetonationPdu):
                    log_entry["firing_entity_id"] = {
                        "site": pdu.firingEntityID.siteID,
                        "application": pdu.firingEntityID.applicationID,
                        "entity": pdu.firingEntityID.entityID
                    }
                    log_entry["target_entity_id"] = {
                        "site": pdu.targetEntityID.siteID,
                        "application": pdu.targetEntityID.applicationID,
                        "entity": pdu.targetEntityID.entityID
                    }
                    log_entry["event_id"] = {
                        "event_id": pdu.eventID
                    }
                    log_entry["location"] = {
                        "x": pdu.location.x,
                        "y": pdu.location.y,
                        "z": pdu.location.z
                    }
                    log_entry["velocity"] = {
                        "x": pdu.velocity.x,
                        "y": pdu.velocity.y,
                        "z": pdu.velocity.z
                    }
                    log_entry["detonation_result"] = pdu.detonationResult
                
                elif isinstance(pdu, CollisionPdu):
                    log_entry["issuing_entity_id"] = {
                        "site": pdu.issuingEntityID.siteID,
                        "application": pdu.issuingEntityID.applicationID,
                        "entity": pdu.issuingEntityID.entityID
                    }
                    log_entry["colliding_entity_id"] = {
                        "site": pdu.collidingEntityID.siteID,
                        "application": pdu.collidingEntityID.applicationID,
                        "entity": pdu.collidingEntityID.entityID
                    }
                    log_entry["event_id"] = {
                        "site": pdu.collidingEntityID.siteID,
                        "application": pdu.collidingEntityID.applicationID,
                    }
                    log_entry["collision_type"] = pdu.collisionType
                    log_entry["velocity"] = {
                        "x": pdu.velocity.x,
                        "y": pdu.velocity.y,
                        "z": pdu.velocity.z
                    }
                    log_entry["mass"] = pdu.mass
                    log_entry["location"] = {
                        "x": pdu.location.x,
                        "y": pdu.location.y,
                        "z": pdu.location.z
                    }
                
                elif isinstance(pdu, SignalPdu):
                    log_entry["entity_id"] = {
                        "site": pdu.entityID.siteID,
                        "application": pdu.entityID.applicationID,
                        "entity": pdu.entityID.entityID
                    }
                    log_entry["radio_id"] = pdu.radioID
                    log_entry["encoding_scheme"] = pdu.encodingScheme
                    log_entry["tdl_type"] = pdu.tdlType
                    log_entry["sample_rate"] = pdu.sampleRate
                    log_entry["samples"] = pdu.samples
                    log_entry["data_length"] = len(pdu.data) if pdu.data else 0
            
            # Write to log file
            def default_serializer(obj):
                return str(obj)

            self.log_file.write(json.dumps(log_entry, default=default_serializer) + "\n")
            self.log_file.flush()
            
        except Exception as e:
            print(f"Error processing PDU: {e}")
            traceback.print_exc()
            
            # Log the error
            if self.log_file:
                error_entry = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "pdu_number": self.pdu_count,
                    "source_ip": addr[0],
                    "source_port": addr[1],
                    "size_bytes": len(data),
                    "raw_data_hex": binascii.hexlify(data).decode('utf-8'),
                    "error": str(e)
                }
                self.log_file.write(json.dumps(error_entry) + "\n")
                self.log_file.flush()

def print_pdu_info(pdu):
    """
    Utility function to recursively print information about a PDU's attributes
    """
    # Get all attributes that don't start with '__'
    attributes = [attr for attr in dir(pdu) if not attr.startswith('__') and not callable(getattr(pdu, attr))]
    
    for attr in attributes:
        value = getattr(pdu, attr)
        
        # Skip internal attributes or functions
        if attr in ['encoderClass', 'serialize', 'parse', 'unMarshal', 'marshal', 'reflection']:
            continue
        
        # If the attribute is an object with its own attributes, recurse
        if hasattr(value, '__dict__') and not isinstance(value, (int, float, str, bool, list, dict)):
            print(f"  {attr}:")
            try:
                print_pdu_info(value)
            except:
                print(f"    {value}")
        else:
            print(f"  {attr}: {value}")

def setup_arg_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='DIS PDU Receiver and Logger')
    
    parser.add_argument('-p', '--port', type=int, default=3000,
                        help='UDP port to listen on (default: 3000)')
    parser.add_argument('-m', '--multicast', type=str, default="127.0.0.1",
                        help='Multicast group to join (default: 127.0.0.1)')
    parser.add_argument('-d', '--directory', type=str, default="pdu_logs",
                        help='Directory to store logs (default: pdu_logs)')
    parser.add_argument('-t', '--timeout', type=float, default=None,
                        help='Timeout in seconds (default: run indefinitely)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Disable console output')
    parser.add_argument('-b', '--no-binary', action='store_true',
                        help='Disable saving binary PDUs')
    parser.add_argument('-r', '--no-raw', action='store_true',
                        help='Disable printing raw hex data to console')
    
    return parser

def main():
    """Main function"""
    print("DIS PDU Receiver and Logger")
    print("===========================")
    
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Create the receiver
    receiver = PDUReceiver(
        port=args.port,
        multicast_group=args.multicast,
        log_dir=args.directory,
        console_output=not args.quiet,
        save_binary=not args.no_binary,
        raw_hex=not args.no_raw
    )
    
    # Start receiving
    try:
        receiver.receive_and_log(args.timeout)
        # check for text file that has been generated and convert to parquet file
        log_files = [f for f in os.listdir(receiver.log_dir) if f.endswith('.txt')]
        if log_files:
            for log_file in log_files:
                log_path = os.path.join(receiver.log_dir, log_file)
                df = pd.read_json(log_path, lines=True)
                parquet_path = os.path.splitext(log_path)[0] + '.parquet'
                df.to_parquet(parquet_path, index=False)
                print(f"Converted {log_path} to {parquet_path}")
        else:
            print("No log files found.")
            
        # convert txt files to pickle files
        pickle_files = [f for f in os.listdir(receiver.log_dir) if f.endswith('.txt')]
        if pickle_files:
            for pickle_file in pickle_files:
                pickle_path = os.path.join(receiver.log_dir, pickle_file)
                df = pd.read_json(pickle_path, lines=True)
                pickle_path = os.path.splitext(pickle_path)[0] + '.pkl'
                df.to_pickle(pickle_path)
                print(f"Converted {pickle_path} to {pickle_path}")
        else:
            print("No pickle files found.")
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        receiver.close()
        print("Receiver closed. Exiting.")

if __name__ == "__main__":
    main()