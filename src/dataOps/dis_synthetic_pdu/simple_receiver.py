#!/usr/bin/env python3

import socket
import struct
import binascii
import time
import datetime
import json
import os
from io import BytesIO

# Import OpenDIS classes
from opendis.dis7 import *
from opendis.PduFactory import createPdu
from opendis.DataInputStream import DataInputStream

def unicast_receiver(port=3000, log_dir="unicast_logs"):
    """A simplified DIS PDU receiver that works with unicast"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"pdu_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    log_file = open(log_file_path, "w")
    
    # Create the UDP socket for unicast
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    
    # Allow reuse of addresses and ports
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to the DIS port
    sock.bind(('0.0.0.0', port))  # Bind to all interfaces
    
    print(f"PDU Unicast Receiver listening on port {port}")
    print(f"Logging to: {log_file_path}")
    print("Waiting for PDUs... Press Ctrl+C to stop.")
    
    # Receive loop
    count = 0
    try:
        while True:
            # Receive data
            data, addr = sock.recvfrom(4096)
            timestamp = datetime.datetime.now()
            count += 1
            
            # Print basic info
            print(f"\n[{timestamp}] Received PDU #{count} from {addr[0]}:{addr[1]} ({len(data)} bytes)")
            
            # Try to parse the PDU
            try:
                # Create PDU using PduFactory
                pdu = createPdu(data)
                
                # Print PDU type and key details
                if pdu is not None:
                    print(f"PDU Type: {pdu.__class__.__name__}")
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
                    elif isinstance(pdu, DetonationPdu):
                        print(f"  Firing Entity ID: {pdu.firingEntityID.entityID}")
                        print(f"  Target Entity ID: {pdu.targetEntityID.entityID}")
                        print(f"  Detonation Result: {pdu.detonationResult}")
                    elif isinstance(pdu, CollisionPdu):
                        print(f"  Issuing Entity ID: {pdu.issuingEntityID.entityID}")
                        print(f"  Colliding Entity ID: {pdu.collidingEntityID.entityID}")
                
                # Log to file
                log_entry = {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "pdu_number": count,
                    "source_ip": addr[0],
                    "source_port": addr[1],
                    "size_bytes": len(data),
                    "raw_data_hex": binascii.hexlify(data).decode('utf-8'),
                    "pdu_type": pdu.__class__.__name__ if pdu else "Unknown"
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                    
            except Exception as e:
                print(f"Error parsing PDU: {e}")
                print(f"First 32 bytes: {binascii.hexlify(data[:32]).decode()}")
            
            
    except KeyboardInterrupt:
        print("\nReceiver stopped by user")
    finally:
        sock.close()
        log_file.close()
        print(f"Received {count} PDUs total")
        print(f"Log saved to {log_file_path}")

if __name__ == "__main__":
    unicast_receiver()