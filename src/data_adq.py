import tkinter as tk
from tkinter import messagebox
from pylsl import StreamInlet, resolve_stream
import threading
import csv
import time

class DataAcquisitionProtocol:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Acquisition Protocol")

        self.label = tk.Label(root, text="Press 'Start' to begin", font=("Helvetica", 16))
        self.label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start", command=self.start_protocol, font=("Helvetica", 14))
        self.start_button.pack(pady=20)

        self.streams = resolve_stream('type', 'EEG')  # Adjust the stream type according to your setup
        self.inlet = StreamInlet(self.streams[0])
        self.data = []

    def start_protocol(self):
        self.start_button.pack_forget()
        self.thread = threading.Thread(target=self.run_protocol)
        self.thread.start()

    def run_protocol(self):
        self.run_module("Rest Module", 20, 'rest_data.csv')
        self.run_rm_mi_module("RM Module", 'rm_data.csv')
        self.run_rm_mi_module("MI Module", 'mi_data.csv')
        messagebox.showinfo("Info", "Protocol Finished")
        self.label.config(text="Protocol Finished")

    def run_module(self, module_name, duration, file_name):
        self.label.config(text=f"{module_name} - {duration} seconds")
        self.wait_for_spacebar()
        self.collect_data(duration, file_name)

    def run_rm_mi_module(self, module_name, file_name):
        subprotocols = [("Flex", 5), ("Rest", 5), ("Extend", 5), ("Rest", 5)]
        for arm in ["Left arm", "Right arm"]:
            for _ in range(10):
                for subprotocol_name, subprotocol_duration in subprotocols:
                    self.label.config(text=f"{module_name} - {arm} - {subprotocol_name} - {subprotocol_duration} seconds")
                    self.wait_for_spacebar()
                    self.collect_data(subprotocol_duration, file_name)

    def wait_for_spacebar(self):
        self.space_pressed = False
        self.root.bind("<space>", self.on_space_press)
        while not self.space_pressed:
            self.root.update()

    def on_space_press(self, event):
        self.space_pressed = True

    def collect_data(self, duration, file_name):
        start_time = time.time()
        while time.time() - start_time < duration:
            sample, timestamp = self.inlet.pull_sample()
            self.data.append([timestamp] + sample)
            time.sleep(0.1)  # Adjust the sleep time if necessary

        self.save_data(file_name)

    def save_data(self, file_name):
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)
        self.data = []

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAcquisitionProtocol(root)
    root.mainloop()
