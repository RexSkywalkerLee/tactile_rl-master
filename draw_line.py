import matplotlib.pyplot as plt
import numpy as np


def draw_dots(sensor_array, ts, y, text=''):
	last_activate_t = -1

	for val, t in zip(sensor_array, ts):
		if val == 0:
			# signal vanished.
			last_activate_t = -1
			continue

		if last_activate_t < 0:
			x1 = [t, t]
			y1 = [y, y]
			last_activate_t = t
			plt.plot(x1, y1, c='r', marker='o')
		else:
			x1 = [last_activate_t, t]
			y1 = [y, y]
			plt.plot(x1, y1, c='r', marker='o')

	if len(text) > 0:
		plt.text(-1.5, y + 0.3, text)

plt.xlim(0, 10)

sensor_array = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
t = np.linspace(0, 10, len(sensor_array))

for i in range(10):
	# draw_dots(sensor_array, t, 1)
	draw_dots(sensor_array, t, i, 'sensor_{}'.format(i))
plt.show()
