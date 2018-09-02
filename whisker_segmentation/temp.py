import matplotlib.pyplot as plt


plt.ioff() # turn off interactive mode to prevent figures from displaying
x, y = (50,100)

fig = plt.figure(frameon=False)
plt.imshow(frame, 'gray')
plt.axis('off')
plt.plot(x, y, 'o', ms=10, alpha=0.8, color=(1,0,0))
fig.canvas.draw()

data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
w, h = fig.canvas.get_width_height()
data = data.reshape((h, w, 3))
plt.show()