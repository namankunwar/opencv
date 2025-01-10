import numpy as np

# Provided coordinates
coordinates = [
    (0.0, 0.0), (1.7241379310344827, 0.0), (3.4482758620689653, 0.0), 
    (5.172413793103448, 0.0), (6.896551724137931, 0.0), (8.620689655172413, 0.0), 
    (10.344827586206897, 0.0), (12.068965517241379, 0.0), (13.793103448275861, 0.0), 
    (15.517241379310343, 0.0), (17.241379310344826, 0.0), (18.96551724137931, 0.0), 
    (20.689655172413794, 0.0), (22.413793103448274, 0.0), (24.137931034482758, 0.0), 
    (25.86206896551724, 0.0), (27.586206896551722, 0.0), (29.310344827586206, 0.0), 
    (31.034482758620687, 0.0), (32.75862068965517, 0.0), (34.48275862068965, 0.0), 
    (36.206896551724135, 0.0), (37.93103448275862, 0.0), (39.6551724137931, 0.0), 
    (41.37931034482759, 0.0), (43.103448275862064, 0.0), (44.82758620689655, 0.0), 
    (46.55172413793103, 0.0), (48.275862068965516, 0.0), (50.0, 0.0), (50.0, 0.0), 
    (51.724137931034484, 0.29726516052318713), (53.44827586206897, 1.1890606420927485), 
    (55.172413793103445, 2.6753864447086766), (56.89655172413793, 4.756242568370984), 
    (58.62068965517241, 7.431629013079666), (60.3448275862069, 10.70154577883472), 
    (62.06896551724138, 14.56599286563615), (63.79310344827586, 19.024970273483937), 
    (65.51724137931035, 24.078478002378134), (67.24137931034483, 29.726516052318665), 
    (68.9655172413793, 35.96908442330556), (70.6896551724138, 42.80618311533888), 
    (72.41379310344827, 50.237812128418526), (74.13793103448276, 58.2639714625446), 
    (75.86206896551724, 66.88466111771699), (77.58620689655172, 76.09988109393575), 
    (79.3103448275862, 85.90963139120095), (81.03448275862068, 96.31391200951245), 
    (82.75862068965517, 107.3127229488704), (84.48275862068965, 118.90606420927466), 
    (86.20689655172413, 131.09393579072525), (87.93103448275862, 143.87633769322235), 
    (89.65517241379311, 157.25326991676582), (91.37931034482759, 171.22473246135553), 
    (93.10344827586206, 185.79072532699163), (94.82758620689654, 200.9512485136741), 
    (96.55172413793103, 216.70630202140308), (98.27586206896552, 233.0558858501784), 
    (100.0, 250.0)
 ]

# Step 1: Separate x and y values
x_vals = np.array([coord[0] for coord in coordinates])
y_vals = np.array([coord[1] for coord in coordinates])

# Step 2: Find xmin, xmax, ymin, ymax
xmin, xmax = x_vals.min(), x_vals.max()
ymin, ymax = y_vals.min(), y_vals.max()

# Step 3: Identify y-values at xmin and xmax
y_at_xmin = y_vals[np.where(x_vals == xmin)[0][0]]
y_at_xmax = y_vals[np.where(x_vals == xmax)[0][0]]

# Step 4: Check trends
if y_at_xmin < y_at_xmax:
    if y_vals[-1] > y_vals[0]:
        result = "Top-right  Curve"
    else:
        result = "Bottom-Left (L) Curve"
else:
    if y_vals[-1] > y_vals[0]:
        result = "Top-left"
    else:
        result = "Bottom-Right (⅃) Curve"

# Step 5: Output
print(f"Detected Curve Shape: {result}")