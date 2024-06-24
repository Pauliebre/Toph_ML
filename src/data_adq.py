import tkinter as tk
from tkinter import simpledialog
import csv
import pylsl
from pylsl import StreamInlet, resolve_stream
import time
from datetime import datetime
import os
import threading

class CountdownApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Data Acquisition Protocol - Motor Imagery")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<space>", self.spacebar_pressed)
        
        # Center the root window
        self.center_window(root)

        # Contador de ciclos
        self.cycle_count = 1
        self.max_cycles = 10

        # Prompts for user input
        self.user_info = self.get_user_info()
        
        # File path for CSV
        self.csv_filename = f"{self.user_info['name']}_{self.user_info['type_of_test']}.csv"
        self.csv_file_path = os.path.join(os.getcwd(), self.csv_filename)
        
        # Create and open the CSV file for writing
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        lsl_headers = [f'Delta {i}' for i in range(1, 9)] + \
                      [f'Theta {i}' for i in range(1, 9)] + \
                      [f'Alpha {i}' for i in range(1, 9)] + \
                      [f'Beta {i}' for i in range(1, 9)] + \
                      [f'Gamma {i}' for i in range(1, 9)]
        self.csv_writer.writerow(['Timestamp', 'Cycle', 'Countdown Type'] + lsl_headers)

        # GUI elements
        self.cycle_label = tk.Label(root, text=f"Cycle: {self.cycle_count}/{self.max_cycles}", font=("Century Gothic", 14))
        self.cycle_label.pack(anchor='ne', padx=10, pady=10)

        self.legend_label = tk.Label(root, text="Relax your muscles, try to think about your tongue", font=("Century Gothic", 44))
        self.legend_label.pack(pady=50)

        self.label = tk.Label(root, text="20", font=("Century Gothic", 150))
        self.label.pack(pady=20)
        
        self.start_button = tk.Button(root, text="Start Rest Module", command=self.start_countdown, font=("Century Gothic", 14))
        self.start_button.pack(pady=60)

        # Inicializa las referencias de los botones
        self.second_button = None
        self.third_button = None
        self.fourth_button = None
        self.fifth_button = None

        # Inicializa las variables para la recolección de datos
        self.current_countdown_type = None
        self.inlet = None
        self.setup_lsl()

        # Thread for collecting data
        self.running = True
        self.data_thread = threading.Thread(target=self.data_collection_thread)
        self.data_thread.start()

    def center_window(self, win):
        win.update_idletasks()
        width = win.winfo_width()
        height = win.winfo_height()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f'{width}x{height}+{x}+{y}')

    def get_user_info(self):
        user_info = {}
        user_info['name'] = simpledialog.askstring("Input", "Enter your name:", parent=self.root)
        user_info['age'] = simpledialog.askinteger("Input", "Enter your age:", parent=self.root)
        user_info['type_of_test'] = simpledialog.askstring("Input", "Enter the type of test:", parent=self.root)
        user_info['dominant_hand'] = simpledialog.askstring("Input", "Enter your dominant hand:", parent=self.root)
        return user_info

    def setup_lsl(self):
        # Resolver el flujo de datos LSL
        print("Looking for an LSL stream...")
        brain_stream = pylsl.resolve_stream("name", "AURA_Power")

        self.inlet = StreamInlet(brain_stream[0])
        info = self.inlet.info()
        sample_rate = info.nominal_srate()
        print(f"LSL stream found with sample rate: {sample_rate} Hz")

    def data_collection_thread(self):
        while self.running:
            if self.current_countdown_type:
                sample, timestamp = self.inlet.pull_sample()
                self.csv_writer.writerow([datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'), 
                                          self.cycle_count, self.current_countdown_type] + sample)
            time.sleep(0.1)  # Collect data 10 times per second

    def start_countdown(self):
        self.current_countdown_type = "First Countdown"
        self.start_button.pack_forget()
        self.countdown(5)

    def start_second_countdown(self):
        self.current_countdown_type = "Second Countdown"
        self.legend_label.config(text="Flex your left arm")
        if self.second_button:
            self.second_button.pack_forget()
        self.countdown(5)

    def start_third_countdown(self):
        self.current_countdown_type = "Third Countdown"
        self.legend_label.config(text="Extend your left arm")
        if self.third_button:
            self.third_button.pack_forget()
        self.countdown(5)

    def start_fourth_countdown(self):
        self.current_countdown_type = "Fourth Countdown"
        self.legend_label.config(text="Flex your right arm")
        if self.fourth_button:
            self.fourth_button.pack_forget()
        self.countdown(5)

    def start_fifth_countdown(self):
        self.current_countdown_type = "Fifth Countdown"
        self.legend_label.config(text="Extend your right arm")
        if self.fifth_button:
            self.fifth_button.pack_forget()
        self.countdown(5)

    def countdown(self, count):
        self.label.config(text=str(count))
        if count > 0:
            self.root.after(1000, self.countdown, count - 1)
        else:
            self.current_countdown_type = None
            if self.second_button is None:
                self.second_button = tk.Button(self.root, text="Start Left Arm Flexion", command=self.start_second_countdown, font=("Century Gothic", 14))
                self.second_button.pack(pady=20)
            elif self.third_button is None:
                self.third_button = tk.Button(self.root, text="Start Left Arm Extension", command=self.start_third_countdown, font=("Century Gothic", 14))
                self.third_button.pack(pady=20)
            elif self.fourth_button is None:
                self.fourth_button = tk.Button(self.root, text="Start Right Arm Flexion", command=self.start_fourth_countdown, font=("Century Gothic", 14))
                self.fourth_button.pack(pady=20)
            elif self.fifth_button is None:
                self.fifth_button = tk.Button(self.root, text="Start Right Arm Extension", command=self.start_fifth_countdown, font=("Century Gothic", 14))
                self.fifth_button.pack(pady=20)
            else:
                self.cycle_count += 1
                self.cycle_label.config(text=f"Cycle: {self.cycle_count}/{self.max_cycles}")
                if self.cycle_count <= self.max_cycles:
                    self.legend_label.config(text=f"Cycle {self.cycle_count-1} Completed! .\nThe cycle will restart in 20 seconds.\nRest your muscles and think of your tongue.")
                    self.countdown_restart(1)
                else:
                    self.legend_label.config(text="All cycles completed!")
                    self.csv_file.close()
                    self.running = False  # Stop the data collection thread

    def countdown_restart(self, count):
        self.label.config(text=str(count))
        if count > 0:
            self.root.after(1000, self.countdown_restart, count - 1)
        else:
            self.repeat_cycle()

    def repeat_cycle(self):
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

    def spacebar_pressed(self, event):
        if self.start_button.winfo_ismapped():
            self.start_countdown()
        elif self.second_button and self.second_button.winfo_ismapped():
            self.start_second_countdown()
        elif self.third_button and self.third_button.winfo_ismapped():
            self.start_third_countdown()
        elif self.fourth_button and self.fourth_button.winfo_ismapped():
            self.start_fourth_countdown()
        elif self.fifth_button and self.fifth_button.winfo_ismapped():
            self.start_fifth_countdown()


    def exit_fullscreen(self, event=None):
        self.csv_file.close()
        self.running = False  # Stop the data collection thread
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = CountdownApp(root)
    root.mainloop()
