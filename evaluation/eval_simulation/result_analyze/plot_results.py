import numpy as np
from matplotlib import pyplot as plt
import pickle


def data_analyze(data):
	run_num = len(data["final_state"])
	outcome = [0, 0, 0]
	path_dis = np.zeros(run_num)
	path_time = np.zeros(run_num)
	for r in range(run_num):
		if data["final_state"][r] == 1 or data["final_state"][r] == 3:
			tmp_overll_path_dis = 0
			for d in range(len(data["robot_path"][r]) - 1):
				rob_pos = data["robot_path"][r][d]
				next_rob_pos = data["robot_path"][r][d + 1]
				tmp_dis = np.sqrt((next_rob_pos[0] - rob_pos[0]) ** 2 + (next_rob_pos[1] - rob_pos[1]) ** 2)
				tmp_overll_path_dis += tmp_dis
			path_dis[r] = tmp_overll_path_dis
			path_time[r] = data["time"][r]
		if data["final_state"][r] == 1:
			outcome[0] += 1
		elif data["final_state"][r] == 2:
			outcome[1] += 1
		elif data["final_state"][r] == 3:
			outcome[2] += 1
		else:
			print("FINAL STATE TYPE ERROR ...")
	return run_num, outcome, path_dis, path_time


def phddpg_average():
	outcome_simp = np.zeros((5, 3))
	outcome = np.zeros((5, 3))
	dist = np.zeros((2, 5))
	time = np.zeros((2, 5))
	for i in range(5):
		FILE_NAME1 = 'pH-DDPG-' + str(i+1) + '_0_199_simple.p'
		FILE_NAME2 = 'pH-DDPG-' + str(i+1) + '_0_199.p'
		run_data1 = pickle.load(open('../record_data/' + FILE_NAME1, 'rb'))
		run_data2 = pickle.load(open('../record_data/' + FILE_NAME2, 'rb'))
		run_num1, outcome1, path_dist1, path_time1 = data_analyze(run_data1)
		run_num2, outcome2, path_dist2, path_time2 = data_analyze(run_data2)
		outcome_simp[i, :] = np.array([outcome1[0], outcome1[1], outcome1[2]])
		outcome[i, :] = np.array([outcome2[0], outcome2[1], outcome2[2]])
		dist1 = np.mean(path_dist1[path_dist1 > 0])
		dist2 = np.mean(path_dist2[path_dist2 > 0])
		dist[:, i] = np.array([dist1, dist2])
		time1 = np.mean(path_time1[path_dist1 > 0])
		time2 = np.mean(path_time2[path_dist2 > 0])
		time[:, i] = np.array([time1, time2])

	avg_outcome_simp = np.mean(outcome_simp, axis=0)
	avg_outcome = np.mean(outcome, axis=0)
	avg_dist = np.mean(dist, axis=1)
	avg_time = np.mean(time, axis=1)
	return avg_outcome_simp, avg_outcome, avg_dist, avg_time


def plot_outcome_rate(methods, data, figlabel):
	data_cum = data.cumsum(axis=1)
	colors = plt.get_cmap('RdYlGn')(np.linspace(0.73, 0.3, data.shape[1]))

	# Figure Size
	fig, ax = plt.subplots(figsize=(10, 6))

	ax.set_xlim(45, 100)

	# Remove axes splines
	for s in ['top', 'bottom', 'right']:
		ax.spines[s].set_visible(False)
	ax.spines['left'].set_linewidth(2)
	# Remove x, y Ticks
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')

	# Add x, y gridlines
	ax.grid(visible=True, color='grey',
		linestyle='-', linewidth=1, alpha=0.2, zorder=0)
	ax.tick_params(axis='y', labelsize=22)
	ax.tick_params(axis='x', labelsize=22)

	# Horizontal Bar Plot
	for i, (colname, color) in enumerate(zip(methods, colors)):
		widths = data[:, i]
		starts = data_cum[:, i] - widths
		ax.barh(methods, widths, left=starts, height=0.42,
				label=colname, color=color, zorder=2)
		xcenters = starts + widths * 0.8

		r, g, b, _ = color
		text_color = 'white' if r * g * b < 0.5 else 'dimgray'
		for y, (x, c) in enumerate(zip(xcenters, widths)):
			ax.text(x, y, str(c), ha='center', va='center', color=text_color,
					fontweight='bold', fontsize=20)

	# rotate ylabel
	labels = ax.get_yticklabels()
	plt.setp(labels, rotation=60, horizontalalignment='right')

	if figlabel == '1':
		ax.legend(labels=['Success', 'Collision', 'Overtime'], ncol=3, bbox_to_anchor=(0, 1),
			  loc='lower left', fontsize=20, edgecolor='white')


def plot_dist_time(methods, data, CI_width, type):
	# set width of bar
	barWidth = 0.25
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.grid(visible=True, color='dimgrey',
			linestyle='-', linewidth=1, alpha=0.2, zorder=0)
	# Remove x, y Ticks
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	# Remove axes splines
	for s in ['top', 'bottom', 'left', 'right']:
		ax.spines[s].set_visible(False)

	# Set position of bar on X axis
	br1 = np.arange(data.shape[1])
	print('br1 = ', br1)
	br2 = [x+barWidth+0.02 for x in br1]
	print('br2 = ', br2)

	if type == 'dist':
		# Make distance plot
		ax.bar(br1, data[0, :], yerr=CI_width[0, :], color='lightskyblue', width=barWidth, zorder=2)
		ax.bar(br2, data[1, :], yerr=CI_width[1, :], color='cornflowerblue', width=barWidth, zorder=2)
		ax.set_ylabel('Route Distance (m)', fontsize=25)
		ax.legend(labels=['Target Motion Policy 1 (Simpler)', 'Target Motion Policy 2 (Harder)'], ncol=3,
				  bbox_to_anchor=(0, 1),
				  loc='lower left', fontsize=20, edgecolor='white')

	elif type == 'time':
		# Make time plot
		ax.bar(br1, data[0, :], yerr=CI_width[0, :], color='lightskyblue', width=barWidth, zorder=2)
		ax.bar(br2, data[1, :], yerr=CI_width[1, :], color='cornflowerblue', width=barWidth, zorder=2)
		ax.set_ylabel('Route time (s)', fontsize=25)

	# Add annotation to bars
	for i in ax.patches:
		plt.text(i.get_x() + (barWidth+0.03)/2, i.get_height() + 0.5,
				 str(round((i.get_height()), 2)),
				 ha='center', fontsize=18,
				 fontweight='bold', color='black')

	# Adding Xtick
	ax.set_xticks([r + 0.5 * (barWidth+0.02) for r in range(data.shape[1])], methods)
	ax.tick_params(axis='x', labelcolor='black', labelsize=22)
	ax.tick_params(axis='y', labelcolor='dimgrey', labelsize=24)


if __name__ == "__main__":
	# method_list = ['pH-DDPG', 'DDPG-GM', 'SP-DDPG', 'DDPG']
	method_list = ['pH-DDPG_disappear', 'DDPG_disappear', 'DDPG_star', 'HR-astw']

	outcome_rate_simp = np.zeros((4, 3))
	outcome_rate = np.zeros((4, 3))
	avg_dist = np.zeros((2, 4))
	avg_time = np.zeros((2, 4))
	dist_CI_width = np.zeros((2, 4))
	time_CI_width = np.zeros((2, 4))
	for i, method in enumerate(method_list):
		if method == 'pH-DDPG':
			# outcome1, outcome2, dist, time = phddpg_average()
			# outcome_rate_simp[0, :] = np.array([outcome1[0], outcome1[1], outcome1[2]]) / 2
			# outcome_rate[0, :] = np.array([outcome2[0], outcome2[1], outcome2[2]]) / 2
			# avg_dist[:, 0] = np.squeeze(dist)
			# avg_time[:, 0] = np.squeeze(time)
			FILE_NAME1 = method + '-1-5_0_999_simple.p'
			FILE_NAME2 = method + '-1-5_0_999.p'
			# FILE_NAME1 = method + '-5_0_199_simple.p'
			# FILE_NAME2 = method + '-5_0_199.p'
		else:
			# FILE_NAME1 = method + '_0_199_simple.p'
			FILE_NAME1 = method + '_0_199.p'

			FILE_NAME2 = method + '_0_199.p'
		run_data1 = pickle.load(open('../record_data/' + FILE_NAME1, 'rb'))
		run_data2 = pickle.load(open('../record_data/' + FILE_NAME2, 'rb'))
		run_num1, outcome1, path_dist1, path_time1 = data_analyze(run_data1)
		run_num2, outcome2, path_dist2, path_time2 = data_analyze(run_data2)

		outcome_rate_simp[i, :] = np.array([outcome1[0], outcome1[1], outcome1[2]])/run_num1 * 100
		outcome_rate[i, :] = np.array([outcome2[0], outcome2[1], outcome2[2]])/run_num2 * 100

		avg_dist1 = np.mean(path_dist1[path_dist1 > 0])
		avg_dist2 = np.mean(path_dist2[path_dist2 > 0])
		avg_dist[:, i] = np.array([avg_dist1, avg_dist2])

		dist_std1 = np.std(path_dist1[path_dist1 > 0])
		print(dist_std1)
		dist_CI_width1 = 1.95996 * dist_std1 / np.sqrt(run_num1)
		dist_std2 = np.std(path_dist2[path_dist2 > 0])
		print(dist_std2)
		dist_CI_width2 = 1.95996 * dist_std2 / np.sqrt(run_num2)
		dist_CI_width[:, i] = np.array([dist_CI_width1, dist_CI_width2])

		avg_time1 = np.mean(path_time1[path_dist1 > 0])
		avg_time2 = np.mean(path_time2[path_dist2 > 0])
		avg_time[:, i] = np.array([avg_time1, avg_time2])

		time_std1 = np.std(path_time1[path_dist1 > 0])
		print(time_std1)
		time_CI_width1 = 1.95996 * time_std1 / np.sqrt(run_num1)
		time_std2 = np.std(path_time2[path_dist2 > 0])
		print(time_std2)
		time_CI_width2 = 1.95996 * time_std2 / np.sqrt(run_num2)
		time_CI_width[:, i] = np.array([time_CI_width1, time_CI_width2])

	# print(outcome_rate_simp)
	# print(outcome_rate)
	print(avg_dist)
	print(avg_time)
	print(dist_CI_width)
	print(time_CI_width)

	plt.rc('font', family='Times New Roman')
	# plot_outcome_rate(method_list, outcome_rate_simp, '1')
	# plot_outcome_rate(method_list, outcome_rate, '2')
	method_list.reverse()
	plot_dist_time(method_list, np.fliplr(avg_dist), np.fliplr(dist_CI_width), 'dist')
	plot_dist_time(method_list, np.fliplr(avg_time), np.fliplr(time_CI_width), 'time')
	plt.show()

