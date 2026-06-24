import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================
csv_path = "cycle_summary.csv"  # one row per cycle

df = pd.read_csv(csv_path)

#check these are proper indicies for cycle, capacity, and CE
cycle_col    = df.columns[0]
capacity_col = df.columns[1]
ce_col       = df.columns[2]

#plotting
print("Columns detected:", df.columns.tolist())
print(f"  Cycle:    '{cycle_col}'")
print(f"  Capacity: '{capacity_col}'")
print(f"  CE:       '{ce_col}'")
print(f"  Cycles:   {df[cycle_col].min()} — {df[cycle_col].max()}")

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.scatter(df[cycle_col], df[capacity_col], color="tab:blue", s=15)
ax1.set_xlabel("Cycle Number")
ax1.set_ylabel(capacity_col, color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.scatter(df[cycle_col], df[ce_col], color="tab:red", s=15)
ax2.set_ylabel(ce_col, color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("Discharge Capacity and Coulombic Efficiency vs. Cycle Number")
fig.tight_layout()
plt.savefig("discharge_ce_plot.png", dpi=150)
plt.show()