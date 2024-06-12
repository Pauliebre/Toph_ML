import serial
import time

# Configuración del puerto serial (ajusta 'COM3' al puerto correcto en tu sistema)
arduino_port = "COM80"  # En Windows normalmente es COMx, en Linux suele ser /dev/ttyUSBx o /dev/ttyACMx
baud_rate = 38400  # Este debe coincidir con la configuración en el Arduino
ser = serial.Serial(arduino_port, baud_rate)

def send_to_arduino(value):
    if value not in [0, 1]:
        print("Solo se puede enviar 0 o 1")
        return
    ser.write(str(value).encode())  # Envía el valor al Arduino

try:
    while True:
        user_input = input("Introduce 0 o 1 para enviar al Arduino ('q' para salir): ")
        if user_input == 'q':
            break
        elif user_input in ['0', '1']:
            send_to_arduino(int(user_input))
        else:
            print("Entrada no válida. Solo se puede enviar 0 o 1.")

except KeyboardInterrupt:
    print("Programa terminado por el usuario.")

finally:
    ser.close()
