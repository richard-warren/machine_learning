

angles = np.genfromtxt(os.path.join('data','raw','frame_angles.csv'))[1:,3]
bins = np.histogram(angles, bins=10)[0]

weight_lims = [.1, 10]
weights = (1/bins) / np.mean(1/bins)
weights = np.clip(weights, weight_lims[0], weight_lims[1])