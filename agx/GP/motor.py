# -*- coding: utf-8 -*-
# pyrefly: ignore [missing-import]
from dynamixel_sdk import *
from time import sleep
from pynput import keyboard
import os

pressed_keys = set()

def on_press(key):
    try:
        if key.char is not None:
            pressed_keys.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        if key.char is not None:
            pressed_keys.discard(key.char)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def is_pressed(key_char):
    return key_char in pressed_keys

    # Initialize PortHandler and PacketHandler
    portHandler   = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(2.0)
    
    portHandler.openPort()
    portHandler.setBaudRate(BAUDRATE)


ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
BAUDRATE = 57600
DEVICENAME = os.getenv(
    "GRIPPER_MOTOR_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7NN-if00-port0",
)

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
DXL_IDS = [1, 2, 3]
# Initialize PortHandler and PacketHandler
portHandler   = PortHandler(DEVICENAME)
packetHandler = PacketHandler(2.0)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

def open_the_gripper():     
    original_pos = [3072,3072,2048]
    sleep(1)
    for idx, dxl_id in enumerate(DXL_IDS):
        packetHandler.write4ByteTxRx(portHandler, dxl_id%3 + 1, ADDR_GOAL_POSITION, original_pos[(idx+1)%3])
        

def control():
    # Reboot motors
    for dxl_id in DXL_IDS:
        packetHandler.reboot(portHandler, dxl_id)
    sleep(1)

    # Enable motor torque
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

    # Read current positions to initialize control variables (avoiding sudden jumps)
    positions = [3072, 3072, 2048]  # Default fallback positions
    for idx, dxl_id in enumerate(DXL_IDS):
        pos, result, err = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS:
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            positions[idx] = pos
            print(f"Motor {dxl_id} initial position: {pos}")
        else:
            print(f"Failed to read Motor {dxl_id} initial position, using default: {positions[idx]}")

    pos1, pos2, pos3 = positions

    print("\nControl Started:")
    print("  Hold 'o' to open gripper (all +15)")
    print("  Hold 'c' to close gripper (all -15)")
    print("  Hold '4'/'1' to open/close motor 1 (+15 / -15)")
    print("  Hold '5'/'2' to open/close motor 2 (+15 / -15)")
    print("  Hold '6'/'3' to open/close motor 3 (+15 / -15)")
    print("  Press '0' to reset position")
    print("  Press 'x' to exit")

    try:
        while True:
            # Control all motors together
            if is_pressed('o'):
                pos1 += 15
                pos2 += 15
                pos3 += 15
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, pos1)
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, pos2)
                packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, pos3)
                sleep(0.02)
                
            elif is_pressed('c'):
                pos1 -= 15
                pos2 -= 15
                pos3 -= 15
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, pos1)
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, pos2)
                packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, pos3)
                sleep(0.02)

            # Control individual motors
            # Motor 1
            if is_pressed('4'):
                pos1 += 15
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, pos1)
                sleep(0.02)
            elif is_pressed('1'):
                pos1 -= 15
                packetHandler.write4ByteTxRx(portHandler, 1, ADDR_GOAL_POSITION, pos1)
                sleep(0.02)

            # Motor 2
            if is_pressed('5'):
                pos2 += 15
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, pos2)
                sleep(0.02)
            elif is_pressed('2'):
                pos2 -= 15
                packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_POSITION, pos2)
                sleep(0.02)

            # Motor 3
            if is_pressed('6'):
                pos3 += 15
                packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, pos3)
                sleep(0.02)
            elif is_pressed('3'):
                pos3 -= 15
                packetHandler.write4ByteTxRx(portHandler, 3, ADDR_GOAL_POSITION, pos3)
                sleep(0.02)

            if is_pressed('0'):
                open_the_gripper()
                pos1, pos2, pos3 = 3072, 3072, 2048
            # Exit control
            if is_pressed('x'):
                print("Exiting...")
                break

    finally:
        # Disable torque and close port
        for dxl_id in DXL_IDS:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        portHandler.closePort()
        print("Motor control port closed.")


def read_motor_positions():

    positions = [3072, 3072, 2048]  
    for idx, dxl_id in enumerate(DXL_IDS):
        pos, result, err = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        if result == COMM_SUCCESS:
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            positions[idx] = pos
    return positions
if __name__ == '__main__':
    control()
