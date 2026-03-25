"""
Updated on Mar 25 2026

@author: Andrew Lawson

MAX31855 Thermocouple Reader
----------------------------

This utility script opens an SPI connection to a MAX31855 thermocouple
interface and continuously reads temperature measurements for basic testing
and diagnostics.

Core responsibilities:
  • Open the configured SPI bus/device
  • Read raw thermocouple data from the MAX31855
  • Convert the reported temperature to Celsius and Fahrenheit
  • Detect basic thermocouple fault conditions
  • Print live temperature readings at a fixed interval

Sensor usage:
  The MAX31855 communicates over SPI and returns a 32-bit data word
  containing:
    - thermocouple temperature
    - fault status
    - auxiliary diagnostic bits

  This script reads that raw value and extracts the thermocouple
  temperature in 0.25 °C increments.

Notes:
  - This script is intended for simple hardware verification and sensor checks.
  - It does not log to file or perform timestamping.
  - If a thermocouple fault is detected, an error is printed instead of a
    temperature value.
  - Readings continue until the user interrupts the program.

Designed for:
  - SPI interface testing
  - thermocouple verification
  - bench diagnostics
  - simple live temperature monitoring
"""

import time
import spidev


# ----------------------------------------
# MAX31855 thermocouple interface
# ----------------------------------------
class MAX31855:
    """
    Simple driver for the MAX31855 thermocouple interface over SPI.

    Provides:
      - raw 32-bit reads
      - Celsius and Fahrenheit temperature conversion
      - basic fault detection
    """
    def __init__(self, bus=0, device=0, max_speed_hz=500000):
        """
        Initialise the SPI connection to the MAX31855.

        Args:
            bus (int): SPI bus number.
            device (int): SPI chip-select device number.
            max_speed_hz (int): SPI clock speed in Hz.
        """
        self.bus = bus
        self.device = device
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed_hz

        # MAX31855 uses SPI mode 0 (CPOL=0, CPHA=0)
        self.spi.mode = 0

    def read_raw(self):
        """
        Read the full 32-bit raw data word from the MAX31855.

        Returns:
            int: Raw 32-bit sensor value.

        Raises:
            RuntimeError: If fewer than 4 bytes are returned.
        """
        raw = self.spi.readbytes(4)
        if len(raw) != 4:
            raise RuntimeError("Failed to read 4 bytes from MAX31855")

        value = (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3]
        return value

    def read_celsius(self):
        """
        Read thermocouple temperature in degrees Celsius.

        Returns:
            float: Temperature in °C.

        Raises:
            RuntimeError: If the MAX31855 reports a fault.
        """
        value = self.read_raw()

        # Fault bit indicates a thermocouple or connection problem
        if (value >> 16) & 0x1:
            raise RuntimeError("Thermocouple fault")

        # Extract signed 14-bit thermocouple temperature
        temp_raw = (value >> 18) & 0x3FFF

        # Convert from signed 14-bit representation if negative
        if temp_raw & 0x2000:
            temp_raw -= 1 << 14

        temp_c = temp_raw * 0.25
        return temp_c

    def read_fahrenheit(self):
        """
        Read thermocouple temperature in degrees Fahrenheit.

        Returns:
            float: Temperature in °F.
        """
        temp_c = self.read_celsius()
        return temp_c * 9.0 / 5.0 + 32.0

    def close(self):
        """Close the SPI device cleanly."""
        self.spi.close()


# ----------------------------------------
# Main live-read loop
# ----------------------------------------
def main():
    """
    Continuously read and print temperature from the MAX31855.

    Behaviour:
      - Opens the SPI-connected thermocouple interface
      - Prints temperature in Celsius and Fahrenheit once per second
      - Prints an error message if a sensor fault is detected
      - Exits cleanly on Ctrl+C
    """
    sensor = MAX31855(bus=0, device=0)  # SPI0, CE0

    try:
        while True:
            try:
                temp_c = sensor.read_celsius()
                temp_f = sensor.read_fahrenheit()
                print(f"Temperature: {temp_c:6.2f} °C  |  {temp_f:6.2f} °F")
            except RuntimeError as e:
                # For example: thermocouple disconnected or fault condition
                print("Error:", e)

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        sensor.close()

if __name__ == "__main__":
    main()