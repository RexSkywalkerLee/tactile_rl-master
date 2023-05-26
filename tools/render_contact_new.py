import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator


def plot_line4(contact):
	print(contact.shape)
	time_scale = len(contact)
	contact_data = contact.T
	x = np.linspace(0, time_scale - 1, time_scale)
	ax = plt.gca()
	ax.yaxis.set_major_locator(MultipleLocator(1))

	plt.xlim(0, 300.1)
	num = 0
	for i in range(16):
		# plt.yticks([])
		if np.sum(contact_data[i]) == 0:
			pass
		if np.sum(contact_data[i]) > 0:
			plt.plot(x, np.clip(contact_data[i], 0, 0.5) + i, linewidth='1.5')
	plt.ylim(-0.5, 15.8)


sns.set(context="notebook", style="whitegrid", palette="deep", font="sans-serif", font_scale=1, color_codes=False,rc=None)

data = np.load('../qpos_trajectory2023-01-30--11-37-58.npy', allow_pickle=True)
# print(data[0]["contact"].shape)#[0]["contact"].shape)
for i in range(8):
	plt.subplot(2, 4, i + 1)
	contact = data[i]["contact"]
	plot_line4(contact)
plt.show()
