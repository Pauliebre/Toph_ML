import tkinter as tk  # Importa la biblioteca Tkinter para crear la GUI

class CountdownApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Data Acquisition Protocol - Motor Imagery")  # Establece el título de la ventana
        self.root.attributes("-fullscreen", True)  # Configura la ventana para pantalla completa
        self.root.bind("<Escape>", self.exit_fullscreen)  # Asocia la tecla Escape para cerrar la aplicación
        
        # Contador de ciclos
        self.cycle_count = 1
        self.max_cycles = 10  # Número máximo de ciclos

        # Crea una etiqueta para mostrar el contador de ciclos
        self.cycle_label = tk.Label(root, text=f"Cycle: {self.cycle_count}/{self.max_cycles}", font=("Century Gothic", 14))
        self.cycle_label.pack(anchor='ne', padx=10, pady=10)  # Muestra la etiqueta en la esquina superior derecha

        # Crea una etiqueta con un mensaje instructivo
        self.legend_label = tk.Label(root, text="Relax your muscles, try to think about your tongue", font=("Century Gothic", 44))
        self.legend_label.pack(pady=50)  # Muestra la etiqueta con un margen superior e inferior de 50 píxeles

        # Crea una etiqueta para mostrar el contador, inicialmente con el valor "20"
        self.label = tk.Label(root, text="20", font=("Century Gothic", 150))
        self.label.pack(pady=20)  # Muestra la etiqueta con un margen superior e inferior de 20 píxeles
        
        # Crea un botón para iniciar el módulo de descanso
        self.start_button = tk.Button(root, text="Start Rest Module", command=self.start_countdown, font=("Century Gothic", 14))
        self.start_button.pack(pady=60)  # Muestra el botón con un margen superior e inferior de 60 píxeles

        # Inicializa las referencias de los botones
        self.second_button = None
        self.third_button = None
        self.fourth_button = None
        self.fifth_button = None

    def start_countdown(self):
        self.start_button.pack_forget()  # Oculta el botón después de iniciar el contador
        self.countdown(20)  # Inicia el contador con 20 segundos

    def start_second_countdown(self):
        self.legend_label.config(text="Flex your left arm")  # Cambia la leyenda al iniciar el contador de 5 segundos
        if self.second_button:
            self.second_button.pack_forget()  # Oculta el segundo botón después de iniciar el contador
        self.countdown(5)  # Inicia el contador con 5 segundos

    def start_third_countdown(self):
        self.legend_label.config(text="Extend your left arm")  # Cambia la leyenda al iniciar el tercer contador
        if self.third_button:
            self.third_button.pack_forget()  # Oculta el tercer botón después de iniciar el contador
        self.countdown(5)  # Inicia el tercer contador con 5 segundos

    def start_fourth_countdown(self):
        self.legend_label.config(text="Flex your right arm")  # Cambia la leyenda al iniciar el cuarto contador
        if self.fourth_button:
            self.fourth_button.pack_forget()  # Oculta el cuarto botón después de iniciar el contador
        self.countdown(5)  # Inicia el cuarto contador con 5 segundos

    def start_fifth_countdown(self):
        self.legend_label.config(text="Extend your right arm")  # Cambia la leyenda al iniciar el quinto contador
        if self.fifth_button:
            self.fifth_button.pack_forget()  # Oculta el quinto botón después de iniciar el contador
        self.countdown(5)  # Inicia el quinto contador con 5 segundos

    def countdown(self, count):
        self.label.config(text=str(count))  # Actualiza el texto de la etiqueta del contador
        if count > 0:
            self.root.after(1000, self.countdown, count - 1)  # Llama a esta función de nuevo después de 1 segundo
        else:
            # Si el contador llega a 0, maneja la lógica de los botones y los contadores
            if count == 0:
                if self.second_button is None and self.cycle_count == 1:
                    # Crea un botón para iniciar el segundo contador de 5 segundos
                    self.second_button = tk.Button(self.root, text="Start Left Arm Flexion", command=self.start_second_countdown, font=("Century Gothic", 14))
                    self.second_button.pack(pady=20)  # Muestra el botón con un margen superior e inferior de 20 píxeles
                elif self.third_button is None:
                    # Crea un botón para iniciar el tercer contador de 5 segundos
                    self.third_button = tk.Button(self.root, text="Start Left Arm Extension", command=self.start_third_countdown, font=("Century Gothic", 14))
                    self.third_button.pack(pady=20)  # Muestra el botón con un margen superior e inferior de 20 píxeles
                elif self.fourth_button is None:
                    # Crea un botón para iniciar el cuarto contador de 5 segundos
                    self.fourth_button = tk.Button(self.root, text="Start Right Arm Flexion", command=self.start_fourth_countdown, font=("Century Gothic", 14))
                    self.fourth_button.pack(pady=20)  # Muestra el botón con un margen superior e inferior de 20 píxeles
                elif self.fifth_button is None:
                    # Crea un botón para iniciar el quinto contador de 5 segundos
                    self.fifth_button = tk.Button(self.root, text="Start Right Arm Extension", command=self.start_fifth_countdown, font=("Century Gothic", 14))
                    self.fifth_button.pack(pady=20)  # Muestra el botón con un margen superior e inferior de 20 píxeles
                else:
                    # Una vez completado el quinto contador, incrementa el número de ciclos y actualiza el contador de ciclos
                    self.cycle_count += 1
                    self.cycle_label.config(text=f"Cycle: {self.cycle_count}/{self.max_cycles}")  # Actualiza el contador de ciclos
                    if self.cycle_count < self.max_cycles:
                        self.legend_label.config(text=f"Cycle {self.cycle_count-1} Completed! .\nThe cycle will restart in 20 seconds.\nRest your muscles and think of your tongue.")
                        self.countdown_restart(1)  # Inicia el contador de 10 segundos antes de repetir el ciclo
                    else:
                        self.legend_label.config(text="All cycles completed!")

    def countdown_restart(self, count):
        self.label.config(text=str(count))  # Actualiza el texto de la etiqueta del contador de reinicio
        if count > 0:
            self.root.after(1000, self.countdown_restart, count - 1)  # Llama a esta función de nuevo después de 1 segundo
        else:
            self.repeat_cycle()

    def repeat_cycle(self):
        # Reinicia los botones y contadores para un nuevo ciclo
        if self.second_button:
            self.second_button.destroy()
            self.second_button = None
        if self.third_button:
            self.third_button.destroy()
            self.third_button = None
        if self.fourth_button:
            self.fourth_button.destroy()
            self.fourth_button = None
        if self.fifth_button:
            self.fifth_button.destroy()
            self.fifth_button = None
        self.start_countdown()

    def exit_fullscreen(self, event=None):
        self.root.quit()  # Cierra la aplicación

if __name__ == "__main__":
    root = tk.Tk()  # Crea la ventana principal de Tkinter
    app = CountdownApp(root)  # Crea una instancia de la aplicación
    root.mainloop()  # Inicia el bucle principal de la aplicación Tkinter
