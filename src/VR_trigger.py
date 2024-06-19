from pylsl import StreamInfo, StreamOutlet
import time

# Crear el stream info.
info = StreamInfo('TriggerStream', 'Markers', 1, 0, 'string', 'myuniquetriggerid')

# Crear el outlet.
outlet = StreamOutlet(info)

# Lista de triggers válidos.
valid_triggers = ["1", "2", "3", "4"]

def send_trigger(trigger):
    if trigger in valid_triggers:
        outlet.push_sample([trigger])
        print(f"Sent trigger: {trigger}")
    else:
        print(f"Invalid trigger: {trigger}. Please enter one of {valid_triggers}.")

while True:
    # Mostrar el menú y capturar la entrada del usuario.
    print("Enter the trigger to send (1, 2, 3, 4) or 'q' to quit:")
    user_input = input().strip()
    
    if user_input.lower() == 'q':
        print("Exiting...")
        break
    else:
        send_trigger(user_input)
        time.sleep(1)  # Espera 1 segundo antes de solicitar nuevamente.
