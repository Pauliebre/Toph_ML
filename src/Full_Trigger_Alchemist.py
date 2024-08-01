import serial
import time
import threading
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

# Configuración de los puertos seriales
arduino_ports = ["COM8", "COM9"]  # Especifica los puertos correctos para tus dispositivos
baud_rate = 38400  # Asegúrate de que coincida con la configuración del Arduino

# Diccionario para almacenar los objetos serial
ser_dict = {}

try:
    for port in arduino_ports:
        ser_dict[port] = serial.Serial(port, baud_rate)
        time.sleep(2)  # Espera 2 segundos para que el Arduino se reinicie

except serial.SerialException as e:
    print(f"Error al abrir el puerto serial: {e}")
    exit()

for port in arduino_ports:
    if not ser_dict[port].is_open:
        print(f"No se pudo abrir el puerto serial {port}.")
        exit()

def send_to_arduino(port, value):
    ser_dict[port].write(str(value).encode())  # Envía el valor al Arduino

# Crear el stream info.
info = StreamInfo('TriggerStream', 'Markers', 1, 0, 'string', 'myuniquetriggerid')

# Crear el outlet.
outlet = StreamOutlet(info)

# Lista de triggers válidos.
valid_triggers = ["0", "1", "2", "3", "4"]

def send_trigger(trigger):
    if trigger in valid_triggers:
        outlet.push_sample([trigger])
        print(f"Sent trigger: {trigger}")
    else:
        print(f"Invalid trigger: {trigger}. Please enter one of {valid_triggers}.")

def handle_lsl_input():
    print("Looking for an LSL stream named 'Riza_Hawkeye'...")
    streams = resolve_stream('name', 'Riza_Hawkeye')
    inlet = StreamInlet(streams[0])

    try:
        while True:
            sample, _ = inlet.pull_sample()
            lsl_input = sample[0]

            if lsl_input in ['0']:
                send_to_arduino("COM8", int(lsl_input))
                send_trigger(lsl_input)
            elif lsl_input in ['2']:
                send_to_arduino("COM9", int(lsl_input) - 2)
                send_trigger(lsl_input)
            elif lsl_input in ['3']:
                send_to_arduino("COM9", int(9))
                send_trigger(lsl_input)
            elif lsl_input in ['1']:
                send_to_arduino("COM8", int(9))
                send_trigger(lsl_input)
            elif lsl_input == '4':
                print("No se envía nada.")
                send_trigger(lsl_input)
            else:
                print("Entrada no válida. Solo se pueden enviar los valores 0, 1, 2, 3 o 4.")
    except KeyboardInterrupt:
        print("Programa terminado por el usuario.")

lsl_thread = threading.Thread(target=handle_lsl_input)
lsl_thread.start()
lsl_thread.join()

for port in arduino_ports:
    if ser_dict[port].is_open:
        ser_dict[port].close()
