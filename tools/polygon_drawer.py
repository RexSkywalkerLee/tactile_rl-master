import matplotlib.pyplot as plt
import matplotlib
import json
import time_utils

ax = plt.figure(figsize=(5, 5))
plt.rcParams['keymap.save'].remove('s')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.ion()

all_polygons = []
past_points = []

# def on_press(event):
# 	print('press', event.key)
# 	if event.key == 's':
# 		print(past_points)
#
# ax.canvas.mpl_connect('key_press_event', on_press)
while len(all_polygons) < 10:
	pts = plt.ginput(1)
	if len(pts) == 0:
		if len(past_points) >= 3:
			# finish drawing and add this.
			all_polygons.append(past_points)

		past_points = []
		plt.cla()
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
		continue

	pt = pts[0]
	plt.scatter([pt[0], pt[0]], [pt[1], pt[1]], c='b')
	if len(past_points) > 0:
		plt.plot([pt[0], past_points[-1][0]], [pt[1], past_points[-1][1]], c='b')


	past_points.append(pt)

with open('rect_{}.txt'.format(time_utils.time_str()), 'w') as f:
	f.write(json.dumps({'rect': all_polygons}))

# import numpy as np
# import matplotlib.pyplot as plt
#
# class DraggableRectangle:
#     def __init__(self, rect):
#         self.rect = rect
#         self.press = None
#
#     def connect(self):
#         """Connect to all the events we need."""
#         self.cidpress = self.rect.figure.canvas.mpl_connect(
#             'button_press_event', self.on_press)
#         self.cidrelease = self.rect.figure.canvas.mpl_connect(
#             'button_release_event', self.on_release)
#         self.cidmotion = self.rect.figure.canvas.mpl_connect(
#             'motion_notify_event', self.on_motion)
#
#     def on_press(self, event):
#         """Check whether mouse is over us; if so, store some data."""
#         if event.inaxes != self.rect.axes:
#             return
#         contains, attrd = self.rect.contains(event)
#         if not contains:
#             return
#         print('event contains', self.rect.xy)
#         self.press = self.rect.xy, (event.xdata, event.ydata)
#
#     def on_motion(self, event):
#         """Move the rectangle if the mouse is over us."""
#         if self.press is None or event.inaxes != self.rect.axes:
#             return
#         (x0, y0), (xpress, ypress) = self.press
#         dx = event.xdata - xpress
#         dy = event.ydata - ypress
#         # print(f'x0={x0}, xpress={xpress}, event.xdata={event.xdata}, '
#         #       f'dx={dx}, x0+dx={x0+dx}')
#         self.rect.set_x(x0+dx)
#         self.rect.set_y(y0+dy)
#
#         self.rect.figure.canvas.draw()
#
#     def on_release(self, event):
#         """Clear button press information."""
#         self.press = None
#         self.rect.figure.canvas.draw()
#
#     def disconnect(self):
#         """Disconnect all callbacks."""
#         self.rect.figure.canvas.mpl_disconnect(self.cidpress)
#         self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
#         self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
#
# fig, ax = plt.subplots()
# rects = ax.bar(range(10), 20*np.random.rand(10))
# drs = []
# for rect in rects:
#     dr = DraggableRectangle(rect)
#     dr.connect()
#     drs.append(dr)
#
# plt.show()