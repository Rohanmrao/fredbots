import tkinter as tk
from tkinter import ttk
import os

# Create the tkinter window
root = tk.Tk()
root.title("Package Assigner")

# Create a frame to hold the input boxes and sliders
frame = ttk.Frame(root, padding=20)
frame.pack()

# Create dropdowns for pickup and destination coordinates
pickup_x_label = ttk.Label(frame, text="Pickup X:")
pickup_x_label.grid(column=0, row=0, padx=10, pady=5)
pickup_x_values = list(range(0, 21))
pickup_x = tk.StringVar(value=pickup_x_values[0])
pickup_x_dropdown = ttk.Combobox(frame, textvariable=pickup_x, values=pickup_x_values, state="readonly")
pickup_x_dropdown.grid(column=1, row=0, padx=10, pady=5)
pickup_x_dropdown.set("Choose Coordinates")

pickup_y_label = ttk.Label(frame, text="Pickup Y:")
pickup_y_label.grid(column=0, row=1, padx=10, pady=5)
pickup_y_values = list(range(0, 21))
pickup_y = tk.StringVar(value=pickup_y_values[0])
pickup_y_dropdown = ttk.Combobox(frame, textvariable=pickup_y, values=pickup_y_values, state="readonly")
pickup_y_dropdown.grid(column=1, row=1, padx=10, pady=5)
pickup_y_dropdown.set("Choose Coordinates")

destination_x_label = ttk.Label(frame, text="Destination X:")
destination_x_label.grid(column=0, row=2, padx=10, pady=5)
destination_x_values = list(range(0, 21))
destination_x = tk.StringVar(value=destination_x_values[0])
destination_x_dropdown = ttk.Combobox(frame, textvariable=destination_x, values=destination_x_values, state="readonly")
destination_x_dropdown.grid(column=1, row=2, padx=10, pady=5)
destination_x_dropdown.set("Choose Coordinates")

destination_y_label = ttk.Label(frame, text="Destination Y:")
destination_y_label.grid(column=0, row=3, padx=10, pady=5)
destination_y_values = list(range(0, 21))
destination_y = tk.StringVar(value=destination_y_values[0])
destination_y_dropdown = ttk.Combobox(frame, textvariable=destination_y, values=destination_y_values, state="readonly")
destination_y_dropdown.grid(column=1, row=3, padx=10, pady=5)
destination_y_dropdown.set("Choose Coordinates")

# Create a label and slider for priority and restrict the slider to integer values between 1 and 10
priority_label = ttk.Label(frame, text="Priority:")
priority_label.grid(column=0, row=4, padx=10, pady=5)
priority_slider = ttk.Scale(frame, from_=1, to=10, orient=tk.HORIZONTAL)
priority_slider.grid(column=1, row=4, padx=10, pady=5)
# add a label to display the current value of the slider
priority_value_label = ttk.Label(frame, text="1")
priority_value_label.grid(column=2, row=4, padx=10, pady=5)
# add a function to update the label when the slider is moved


def update_priority_value_label(event):
    priority_value_label.configure(text=str(int(priority_slider.get())))


priority_slider.bind("<ButtonRelease-1>", update_priority_value_label)

# Create a button to save the values to a text file


def save_values():
    obstacles = [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9],
             [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [
                 8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20],
             [14, 0], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9]]

    if "Choose Coordinates" in [pickup_x.get(), pickup_y.get(), destination_x.get(), destination_y.get()]:
        # show an error message if any of the fields are empty in a new window and place it over the root window
        error_window = tk.Toplevel(root)
        error_window.title("Error")
        error_label = ttk.Label(error_window, text="Please fill out all fields.")

        def close_error_window():
            error_window.destroy()

        close_error_button = ttk.Button(error_window, text="Close", command=close_error_window)
        error_label.pack(padx=10, pady=10)
        close_error_button.pack(padx=10, pady=10)
        error_window.transient(root)
        error_window.grab_set()
        root.wait_window(error_window)
    
    elif [int(pickup_x.get()), int(pickup_y.get())] in obstacles or [int(destination_x.get()), int(destination_y.get())] in obstacles:
        # show an error message if any of the fields are empty in a new window and place it over the root window
        error_window = tk.Toplevel(root)
        error_window.title("Error")
        error_label = ttk.Label(error_window, text="Please choose a different coordinate.")

        def close_error_window():
            error_window.destroy()

        close_error_button = ttk.Button(error_window, text="Close", command=close_error_window)
        error_label.pack(padx=10, pady=10)
        close_error_button.pack(padx=10, pady=10)
        error_window.transient(root)
        error_window.grab_set()
        root.wait_window(error_window)
    else:
        # get the path of the current file
        path = os.path.dirname(os.path.abspath(__file__))
        with open(path + "/packages/package.txt", "a") as f:
            # f.write(f"{package_id.get()} ")
            f.write(f"{pickup_x.get()} ")
            f.write(f"{pickup_y.get()} ")
            f.write(f"{destination_x.get()} ")
            f.write(f"{destination_y.get()} ")
            f.write(f"{int(priority_slider.get())}\n")
        print("Package saved to file.")
    # root.destroy()  # destroy the root window to end the program


save_button = ttk.Button(frame, text="Deploy Package", command=save_values)
save_button.grid(column=1, row=5, padx=10, pady=5)

# # Bind the WM_DELETE_WINDOW protocol to the root window
# root.protocol("WM_DELETE_WINDOW", save_values)

# Start the tkinter event loop
root.mainloop()