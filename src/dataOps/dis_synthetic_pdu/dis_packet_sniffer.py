#!/usr/bin/env python3

"""
DIS Packet Sniffer - Simplified Unicast Version
Receives PDUs via UDP and logs them in CSV/Parquet
"""

import socket
import argparse
import pandas as pd
from datetime import datetime
from opendis.PduFactory import createPdu

PORT = 3000
BUFFER_SIZE = 2048

def receive_and_log_pdus(sock, args):
    logs = []
    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            timestamp = datetime.utcnow()

            try:
                pdu = createPdu(data)  # Directly parse PDU
                print(f"[{timestamp}] Received {pdu.__class__.__name__} from {addr}")
                logs.append({
                    "timestamp": timestamp,
                    "source_ip": addr[0],
                    "source_port": addr[1],
                    "pdu_type": pdu.pduType,
                    "pdu_class": pdu.__class__.__name__,
                    "pdu_hex": data.hex()
                })
            except Exception as e:
                print(f"Failed to parse PDU from {addr}: {e}")

            if args.limit and len(logs) >= args.limit:
                print("Reached limit of captured PDUs.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    if logs:
        df = pd.DataFrame(logs)
        if args.output.endswith(".csv"):
            df.to_csv(args.output, index=False)
        else:
            df.to_parquet(args.output, index=False)
        print(f"Saved {len(logs)} PDUs to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="DIS Packet Sniffer")
    parser.add_argument("--output", default="sniffer_log.parquet", help="Output CSV or Parquet filename")
    parser.add_argument("--limit", type=int, help="Stop after N PDUs")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', PORT))

    print(f"Listening for DIS PDUs on UDP port {PORT}...")
    receive_and_log_pdus(sock, args)

if __name__ == "__main__":
    main()
