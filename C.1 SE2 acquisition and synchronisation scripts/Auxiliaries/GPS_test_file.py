"""
Updated on Mar 25 2026

@author: Andrew Lawson

LC29H GNSS UTC Time Reader
--------------------------

This utility script opens a serial connection to the LC29H GNSS receiver and
continuously monitors incoming NMEA sentences in order to extract valid UTC
date/time information from RMC messages.

Core responsibilities:
  • Open the configured UART serial port at the specified baud rate
  • Read live NMEA output from the LC29H receiver
  • Parse valid $..RMC sentences
  • Extract UTC date and time from RMC messages with valid fix status
  • Print GNSS-derived UTC timestamps in ISO 8601 format

RMC usage:
  RMC ("Recommended Minimum Navigation Information") sentences contain the
  essential navigation/time payload needed here:
    - UTC time
    - fix validity
    - UTC date

  Only RMC sentences with status 'A' (valid fix) are accepted.

Notes:
  - This script does not configure the GNSS module; it only listens to output.
  - No local system time is used for the reported timestamp; only GNSS UTC is used.
  - If no valid fix is available, no timestamp will be printed.
  - This is intended as a lightweight test/diagnostic script for verifying
    GNSS communication and time parsing on the Raspberry Pi.

Designed for:
  - serial link testing
  - GNSS time verification
  - debugging LC29H output
  - lightweight field diagnostics
"""

import serial
from datetime import datetime, timezone
import time

# ----------------------------------------
# Serial port configuration
# ----------------------------------------
# Adjust the serial port and baud rate to match the LC29H connection
PORT = "/dev/ttyS0"
BAUD = 115200


# ----------------------------------------
# NMEA RMC parsing
# ----------------------------------------
def parse_rmc(nmea):
    """
    Parse a $..RMC NMEA sentence and return a UTC datetime if valid.

    Expected fields used:
      - parts[1]: UTC time (hhmmss.sss)
      - parts[2]: fix validity ('A' = valid)
      - parts[9]: UTC date (ddmmyy)

    Example:
        $GNRMC,123519.000,A,....,210225,...*CS

    Args:
        nmea (str): Raw NMEA sentence as a decoded string.

    Returns:
        datetime | None:
            UTC-aware datetime object if parsing succeeds and the fix is valid,
            otherwise None.
    """
    if not nmea or not nmea.startswith("$"):
        return None

    # Strip checksum portion for easier comma-splitting
    nmea_no_cs = nmea.split("*", 1)[0]

    parts = nmea_no_cs.split(",")
    if len(parts) < 10:
        return None

    header = parts[0]
    if len(header) < 6 or header[-3:] != "RMC":
        return None

    time_str = parts[1]
    status = parts[2]
    date_str = parts[9]

    # Only accept valid-fix RMC messages
    if status != "A":
        return None

    # Require at least hhmmss and ddmmyy
    if not time_str or not date_str or len(date_str) < 6 or len(time_str) < 6:
        return None

    try:
        # ----- Time -----
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])

        us = 0
        if "." in time_str:
            frac = time_str.split(".", 1)[1]
            frac = (frac + "000000")[:6]
            us = int(frac)

        # ----- Date -----
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = 2000 + int(date_str[4:6])

        return datetime(year, month, day, hh, mm, ss, us, tzinfo=timezone.utc)

    except Exception:
        return None


# ----------------------------------------
# Main serial read loop
# ----------------------------------------
def main():
    """
    Continuously read NMEA data from the LC29H and print valid GNSS UTC time.

    Behaviour:
      - Opens the configured serial port
      - Clears any buffered startup data
      - Reads line-by-line NMEA output
      - Parses RMC messages only
      - Prints UTC time when a valid fix is present
      - Prints a simple failure message if a read/parsing exception occurs
    """
    print("Reading UTC time from LC29H...\n")

    with serial.Serial(PORT, BAUD, timeout=5) as ser:
        ser.reset_input_buffer()

        while True:
            try:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode(errors="ignore").strip()
                dt = parse_rmc(line)

                if dt:
                    print("GNSS UTC:", dt.isoformat())

            except Exception:
                print("fail to get a reading")
                time.sleep(0.2)


if __name__ == "__main__":
    main()