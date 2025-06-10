import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "/mnt/data/duffing_frf_data.csv"
df = pd.read_csv(file_path)
print("File Read Complete")
# Plot each amplitude column against omega
plt.figure(figsize=(12, 8))
for column in df.columns[1:]:  # Skip the omega column
    plt.plot(df['omega'], df[column], label=column.strip())

plt.xlabel('Omega (rad/s)')
plt.ylabel('Amplitude')
plt.title('Duffing Oscillator Frequency Response')
plt.legend(title='Forcing Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
