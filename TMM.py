import sys
import os

# Redirect stdout to null in frozen mode (executable)
if getattr(sys, 'frozen', False):
    sys.stdout = open(os.devnull, 'w')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from difflib import get_close_matches
from scipy.interpolate import interp1d
from tqdm import tqdm
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
n1 = 1.0  # Air
n2 = 1.6  # Thin film (e.g., MgF2)
n3 = 1.5  # Substrate (e.g., glass)

# Prompt user for CSV file path
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
csv_file_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV files", "*.csv")]
)
if not csv_file_path:
    print("No file selected.")
    exit()

# Remove surrounding quotes
if csv_file_path.startswith('"') and csv_file_path.endswith('"'):
    csv_file_path = csv_file_path[1:-1]
csv_file_path = os.path.normpath(csv_file_path)

if not os.path.exists(csv_file_path):
    print(f"The file {csv_file_path} does not exist.")
    exit()

try:
    df_csv = pd.read_csv(csv_file_path, skiprows=19, header=None, usecols=[0, 1])
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

if df_csv.empty:
    print("The CSV file is empty.")
    exit()

df_csv.columns = ["Wavelength", "Transmittance"]

output_folder = os.path.join(os.getcwd(), "output_files")
os.makedirs(output_folder, exist_ok=True)
output_excel_path = os.path.join(output_folder, "converted_output.xlsx")

try:
    df_csv.to_excel(output_excel_path, index=False)
    print(f"\nCSV file successfully converted to Excel at: {output_excel_path}")
except Exception as e:
    print(f"Error writing Excel file: {e}")
    exit()

# Read Excel
try:
    data = pd.read_excel(output_excel_path, engine='openpyxl')
except Exception as e:
    print(f"An error occurred while reading the Excel file: {e}")
    exit()

data.columns = data.columns.str.strip().str.lower().str.replace(" ", "").str.replace("(", "").str.replace(")", "")
for col in ['wavelength', 'transmittance']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.dropna(subset=['wavelength', 'transmittance'], inplace=True)

required_columns = ['wavelength', 'transmittance']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    for col in missing_columns:
        matches = get_close_matches(col, data.columns, n=1, cutoff=0.6)
        if matches:
            print(f"Did you mean '{matches[0]}' instead of '{col}'?")
    print(f"\nThe Excel file must contain the columns: {required_columns} (case-insensitive).")
    exit()

# Wavelength range input via GUI
min_wavelength_input = simpledialog.askfloat("Wavelength Input", "Enter the minimum wavelength (nm):")
if min_wavelength_input is None:
    print("Minimum wavelength not provided. Exiting.")
    exit()

max_wavelength_input = simpledialog.askfloat("Wavelength Input", "Enter the maximum wavelength (nm):")
if max_wavelength_input is None:
    print("Maximum wavelength not provided. Exiting.")
    exit()

filtered_data = data[(data['wavelength'] >= min_wavelength_input) & (data['wavelength'] <= max_wavelength_input)]
if filtered_data.empty:
    print("No data found in the specified wavelength range.")
    exit()

wavelength_exp_nm = filtered_data['wavelength'].to_numpy()
transmittance_exp_values = filtered_data['transmittance'].to_numpy()
transmittance_exp_values /= np.max(transmittance_exp_values)  # Normalize

# Theoretical wavelength range (for model)
step = 0.5
wavelength_range_nm_theoretical = np.arange(min_wavelength_input, max_wavelength_input + step, step)
wavelength_range_m_theoretical = wavelength_range_nm_theoretical * 1e-9


def calculate_transmittance(thickness):
    k0 = 2 * np.pi / wavelength_range_m_theoretical
    delta = n2 * k0 * thickness

    r01 = (n1 - n2) / (n1 + n2)
    t01 = 2 * n1 / (n1 + n2)
    r12 = (n2 - n3) / (n2 + n3)
    t12 = 2 * n2 / (n2 + n3)

    T = np.zeros_like(wavelength_range_nm_theoretical)

    for i in range(len(wavelength_range_nm_theoretical)):
        P = np.array([[np.exp(-1j * delta[i]), 0],
                      [0, np.exp(1j * delta[i])]])

        M01 = (1 / t01) * np.array([[1, r01],
                                    [r01, 1]])

        M12 = (1 / t12) * np.array([[1, r12],
                                    [r12, 1]])

        M = M01 @ P @ M12

        t_total = 1 / M[0, 0]
        T[i] = (n3 / n1) * np.abs(t_total) ** 2

    return T / np.max(T) if np.max(T) != 0 else T


def calculate_error(thickness):
    T_theoretical = calculate_transmittance(thickness)
    interp_func = interp1d(wavelength_range_nm_theoretical, T_theoretical, kind='cubic', bounds_error=False,
                           fill_value=np.nan)
    T_interpolated = interp_func(wavelength_exp_nm)
    T_interpolated /= np.max(T_interpolated) if np.max(T_interpolated) != 0 else 1
    transmittance_exp_values_normalized = transmittance_exp_values / np.max(transmittance_exp_values)
    correlation = np.corrcoef(T_interpolated, transmittance_exp_values_normalized)[0, 1]
    shape_error = 1 - abs(correlation)
    amplitude_error = np.sqrt(np.mean((T_interpolated - transmittance_exp_values_normalized) ** 2))
    total_error = (0.7 * shape_error) + (0.3 * amplitude_error)
    return total_error


# Thickness optimization
thicknesses = np.linspace(100e-9, 2000e-9, 10000)
best_thickness = None
min_error = float('inf')

for t in tqdm(thicknesses, desc="Optimizing thickness", disable=True):
    error = calculate_error(t)
    if error < min_error:
        min_error = error
        best_thickness = t

if best_thickness is not None:
    print(f"\n Best thickness: {best_thickness * 1e9:.2f} nm")
else:
    print("No valid thickness found.")
    exit()


def create_interactive_gui():
    root = tk.Tk()
    root.title("Thin Film Analysis")
    fig, ax1 = plt.subplots(figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    control_frame = ttk.Frame(root)
    control_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Update plot function
    def update_plot():
        try:
            T_new = calculate_transmittance(best_thickness)
            error = calculate_error(best_thickness)
            ax1.clear()
            ax1.plot(wavelength_range_nm_theoretical, T_new, label=f'Theoretical (t={best_thickness*1e9:.1f} nm)', color='blue')
            ax1.scatter(wavelength_exp_nm, transmittance_exp_values, color='black', label='Experimental', zorder=5)
            ax1.set_title(f"Thickness: {best_thickness*1e9:.1f} nm (Error: {error:.6f})")
            ax1.set_xlabel("Wavelength (nm)")
            ax1.set_ylabel("Normalized Transmittance")
            ax1.set_ylim(0, 1.2)
            ax1.grid(True)
            ax1.legend()
            canvas.draw()
        except Exception as e:
            print(f"Error in update_plot: {e}")

    # Add "Save Graph" button
    def save_graph():
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                fig.savefig(file_path)
                print(f"Graph saved successfully at: {file_path}")
        except Exception as e:
            print(f"Error saving graph: {e}")

    ttk.Button(control_frame, text="Save Graph", command=save_graph).pack(pady=10)

    update_plot()
    root.mainloop()


create_interactive_gui()
