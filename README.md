


# Approach:
The main differentiator of this repository is its RCA (root cause analysis)/ DMI (degradation mode identification) approach/flow of the scripts. Taking you from the simplest to most time consuming analysis to narrow down degradation in electrochemical cells. 
# Logical Flow:
## General Coulometry:
Generating discharge capacity and coulombic efficiency vs. cycle curves requires the most basic of data ($Q=\int I \,dt $) from galvanostatic (constant current) testing. It also can quickly give qualitative hints for the present degradation modes in a cell.
### Interpretation:
| Function | Life Stage | Variable | Behavior | Interpretation |
| :--- | :---: | :---: | :---: | ---: |
| Discharge Capacity | Any | dQ/dCycle | Linear Decline (-c) | Steady-state deg. (i.e. gradual SEI growth) |
| Discharge Capacity | Early | dQ/dCycle | ...
## DVA:
Differential Voltage Analysis (dV/dQ vs. Q) goes hand and hand with the other most common differential analysis tool: Incremental Capacity Analysis (dQ/dV vs. V). DVA enables the detection of redox transitions...
### Savitsky Golay Filtering Advice:
Too small a window leaves noise that obscures real peaks or creates fake ones; too large a window can flatten or merge real, distinct peaks into one blob, or shift their apparent position — which would actively corrupt your LLI/LAM peak-shift comparison between cycle 10 and cycle 40, since the whole point of that comparison depends on accurately tracking small peak position and height changes. There's no universal "correct" window size — it depends on your data's point density (how many V-Q points exist per cycle) and the resolution of features you're trying to preserve. People typically tune it empirically: try a window, check whether real, expected peaks (informed by literature for your chemistry) survive cleanly without spurious wiggles, and adjust.