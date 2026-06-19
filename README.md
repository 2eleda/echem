



## DVA:
### Savitsky Golay Filtering Advice:
Too small a window leaves noise that obscures real peaks or creates fake ones; too large a window can flatten or merge real, distinct peaks into one blob, or shift their apparent position — which would actively corrupt your LLI/LAM peak-shift comparison between cycle 10 and cycle 40, since the whole point of that comparison depends on accurately tracking small peak position and height changes. There's no universal "correct" window size — it depends on your data's point density (how many V-Q points exist per cycle) and the resolution of features you're trying to preserve. People typically tune it empirically: try a window, check whether real, expected peaks (informed by literature for your chemistry) survive cleanly without spurious wiggles, and adjust.