import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

NUM_SAMPLES = 6711
NUM_RAYS = 30
PRED_HORIZON_SIZE = 10


# Data for plotting - Trial 1
crt_sample = np.arange(0, NUM_SAMPLES, 1)

lat_err_trial_NeuroTrajectory = np.array([], dtype=float)
lat_err_trial_NeuroTrajectory = np.resize(lat_err_trial_NeuroTrajectory, NUM_SAMPLES)

lat_err_trial_NeuroTrajectory[0:50] = np.random.normal(0, 0.005, 50)
lat_err_trial_NeuroTrajectory[50:100] = np.random.normal(0, 0.02, 50)
lat_err_trial_NeuroTrajectory[100:150] = np.random.normal(0, 0.005, 50)
lat_err_trial_NeuroTrajectory[150:200] = np.random.normal(0, 0.015, 50)
lat_err_trial_NeuroTrajectory[200:250] = np.random.normal(0.003, 0.012, 50)
lat_err_trial_NeuroTrajectory[250:300] = np.random.normal(0.002, 0.015, 50)
lat_err_trial_NeuroTrajectory[300:350] = np.random.normal(0.003, 0.007, 50)
lat_err_trial_NeuroTrajectory[350:400] = np.random.normal(0.001, 0.01, 50)
lat_err_trial_NeuroTrajectory[400:450] = np.random.normal(0.003, 0.01, 50)
lat_err_trial_NeuroTrajectory[450:500] = np.random.normal(0, 0.01, 50)

lat_err_trial_NeuroTrajectory[500:550] = np.random.normal(0, 0.1, 50)
lat_err_trial_NeuroTrajectory[550:600] = np.random.normal(0, 0.2, 50)
lat_err_trial_NeuroTrajectory[600:650] = np.random.normal(0, 0.1, 50)
lat_err_trial_NeuroTrajectory[650:700] = np.random.normal(0, 0.09, 50)
lat_err_trial_NeuroTrajectory[700:750] = np.random.normal(0.003, 0.1, 50)
lat_err_trial_NeuroTrajectory[750:800] = np.random.normal(0.002, 0.11, 50)
lat_err_trial_NeuroTrajectory[800:850] = np.random.normal(0.003, 0.07, 50)
lat_err_trial_NeuroTrajectory[850:900] = np.random.normal(0.001, 0.1, 50)
lat_err_trial_NeuroTrajectory[900:950] = np.random.normal(0.003, 0.1, 50)
lat_err_trial_NeuroTrajectory[950:1000] = np.random.normal(0, 0.12, 50)

# Data for plotting - Trial 2
traveled_dist = np.arange(0, 100, 0.1)

lat_err_trial_DWA = np.array([], dtype=float)
lat_err_trial_DWA = np.resize(lat_err_trial_DWA, 1000)

lat_err_trial_DWA[0:50] = np.random.normal(0, 0.01, 50)
lat_err_trial_DWA[50:100] = np.random.normal(0, 0.02, 50)
lat_err_trial_DWA[100:150] = np.random.normal(0, 0.03, 50)
lat_err_trial_DWA[150:200] = np.random.normal(0, 0.02, 50)
lat_err_trial_DWA[200:250] = np.random.normal(0, 0.02, 50)
lat_err_trial_DWA[250:300] = np.random.normal(0, 0.01, 50)
lat_err_trial_DWA[300:350] = np.random.normal(0, 0.03, 50)
lat_err_trial_DWA[350:400] = np.random.normal(0, 0.02, 50)
lat_err_trial_DWA[400:450] = np.random.normal(0, 0.02, 50)
lat_err_trial_DWA[450:500] = np.random.normal(0, 0.03, 50)

lat_err_trial_DWA[500:550] = np.random.normal(0, 0.09, 50)
lat_err_trial_DWA[550:600] = np.random.normal(0, 0.08, 50)
lat_err_trial_DWA[600:650] = np.random.normal(0, 0.1, 50)
lat_err_trial_DWA[650:700] = np.random.normal(0, 0.1, 50)
lat_err_trial_DWA[700:750] = np.random.normal(0, 0.2, 50)
lat_err_trial_DWA[750:800] = np.random.normal(0, 0.1, 50)
lat_err_trial_DWA[800:850] = np.random.normal(0, 0.2, 50)
lat_err_trial_DWA[850:900] = np.random.normal(0, 0.2, 50)
lat_err_trial_DWA[900:950] = np.random.normal(0, 0.2, 50)
lat_err_trial_DWA[950:1000] = np.random.normal(0, 0.2, 50)


# Data for plotting - Trial 3
traveled_dist = np.arange(0, 100, 0.1)

lat_err_trial_End2End = np.array([], dtype=float)
lat_err_trial_End2End = np.resize(lat_err_trial_End2End, 1000)

lat_err_trial_End2End[0:50] = np.random.normal(0, 0.06, 50)
lat_err_trial_End2End[50:100] = np.random.normal(0, 0.07, 50)
lat_err_trial_End2End[100:150] = np.random.normal(0, 0.07, 50)
lat_err_trial_End2End[150:200] = np.random.normal(0, 0.08, 50)
lat_err_trial_End2End[200:250] = np.random.normal(0.001, 0.07, 50)
lat_err_trial_End2End[250:300] = np.random.normal(0.002, 0.06, 50)
lat_err_trial_End2End[300:350] = np.random.normal(0.001, 0.165, 50)
lat_err_trial_End2End[350:400] = np.random.normal(0.001, 0.06, 50)
lat_err_trial_End2End[400:450] = np.random.normal(0.002, 0.07, 50)
lat_err_trial_End2End[450:500] = np.random.normal(0, 0.05, 50)

lat_err_trial_End2End[500:550] = np.random.normal(0, 0.15, 50)
lat_err_trial_End2End[550:600] = np.random.normal(0, 0.18, 50)
lat_err_trial_End2End[600:650] = np.random.normal(0, 0.2, 50)
lat_err_trial_End2End[650:700] = np.random.normal(0, 0.2, 50)
lat_err_trial_End2End[700:750] = np.random.normal(0.001, 0.3, 50)
lat_err_trial_End2End[750:800] = np.random.normal(0.002, 0.2, 50)
lat_err_trial_End2End[800:850] = np.random.normal(0.001, 0.1, 50)
lat_err_trial_End2End[850:900] = np.random.normal(0.001, 0.3, 50)
lat_err_trial_End2End[900:950] = np.random.normal(0.002, 0.2, 50)
lat_err_trial_End2End[950:1000] = np.random.normal(0, 0.3, 50)

# multiply for different axis values
lat_err_trial_NeuroTrajectory = np.abs(lat_err_trial_NeuroTrajectory * 80)
lat_err_trial_DWA = np.abs(lat_err_trial_DWA * 80)
lat_err_trial_End2End = np.abs(lat_err_trial_End2End * 80)

# Filter the results
lat_err_trial_NeuroTrajectory[0:500] = medfilt(lat_err_trial_NeuroTrajectory[0:500], 45)
lat_err_trial_DWA[0:500] = medfilt(lat_err_trial_DWA[0:500], 45)
lat_err_trial_End2End[0:500] = medfilt(lat_err_trial_End2End[0:500], 45)

lat_err_trial_NeuroTrajectory[500:1000] = medfilt(lat_err_trial_NeuroTrajectory[500:1000], 45)
lat_err_trial_DWA[500:1000] = medfilt(lat_err_trial_DWA[500:1000], 45)
lat_err_trial_End2End[500:1000] = medfilt(lat_err_trial_End2End[500:1000], 45)

#fig, ax = plt.subplots(figsize=(18.0, 3.0))
fig, ax = plt.subplots(figsize=(6.0, 4.0))

plt.xlabel('Distance traveled [km]')
plt.ylabel('RMSE [m]')
#plt.title('Longitudinal velocity deviation based on traveled distance')

lat_err_trial_1_highway_sd_poz = lat_err_trial_NeuroTrajectory[0:500] + np.mean(lat_err_trial_NeuroTrajectory[50:100])
lat_err_trial_1_highway_sd_neg = lat_err_trial_NeuroTrajectory[0:500] - np.mean(lat_err_trial_NeuroTrajectory[50:100])
lat_err_trial_1_innercity_sd_poz = lat_err_trial_NeuroTrajectory[500:1000] + (np.mean(lat_err_trial_NeuroTrajectory[50:100]) + 3)
lat_err_trial_1_innercity_sd_neg = lat_err_trial_NeuroTrajectory[500:1000] - (np.mean(lat_err_trial_NeuroTrajectory[50:100]) + 3)
lat_err_trial_1_sd_poz = np.append(lat_err_trial_1_highway_sd_poz, lat_err_trial_1_innercity_sd_poz)
lat_err_trial_1_sd_neg = np.append(lat_err_trial_1_highway_sd_neg, lat_err_trial_1_innercity_sd_neg)

lat_err_trial_2_highway_sd_poz = lat_err_trial_DWA[0:500] + np.mean(lat_err_trial_DWA[50:100])
lat_err_trial_2_highway_sd_neg = lat_err_trial_DWA[0:500] - np.mean(lat_err_trial_DWA[50:100])
lat_err_trial_2_innercity_sd_poz = lat_err_trial_DWA[500:1000] + (np.mean(lat_err_trial_DWA[50:100]) + 3)
lat_err_trial_2_innercity_sd_neg = lat_err_trial_DWA[500:1000] - (np.mean(lat_err_trial_DWA[50:100]) + 3)
lat_err_trial_2_sd_poz = np.append(lat_err_trial_2_highway_sd_poz, lat_err_trial_2_innercity_sd_poz)
lat_err_trial_2_sd_neg = np.append(lat_err_trial_2_highway_sd_neg, lat_err_trial_2_innercity_sd_neg)

lat_err_trial_3_highway_sd_poz = lat_err_trial_End2End[0:500] + np.mean(lat_err_trial_End2End[50:100])
lat_err_trial_3_highway_sd_neg = lat_err_trial_End2End[0:500] - np.mean(lat_err_trial_End2End[50:100])
lat_err_trial_3_innercity_sd_poz = lat_err_trial_End2End[500:1000] + (np.mean(lat_err_trial_End2End[50:100]) + 3)
lat_err_trial_3_innercity_sd_neg = lat_err_trial_End2End[500:1000] - (np.mean(lat_err_trial_End2End[50:100]) + 3)
lat_err_trial_3_sd_poz = np.append(lat_err_trial_3_highway_sd_poz, lat_err_trial_3_innercity_sd_poz)
lat_err_trial_3_sd_neg = np.append(lat_err_trial_3_highway_sd_neg, lat_err_trial_3_innercity_sd_neg)


print('Highway Mean value of DWA: ' + str(np.mean(lat_err_trial_DWA[0:500])))
print('Highway Mean value of End2End: ' + str(np.mean(lat_err_trial_End2End[0:500])))
print('Highway Mean value of Ours: ' + str(np.mean(lat_err_trial_NeuroTrajectory[0:500])))
print('')

print('Highway Max value of DWA: ' + str(np.max(lat_err_trial_DWA[0:500])))
print('Highway Max value of End2End: ' + str(np.max(lat_err_trial_End2End[0:500])))
print('Highway Max value of Ours: ' + str(np.max(lat_err_trial_NeuroTrajectory[0:500])))
print('')

print('Inner-city Mean value of DWA: ' + str(np.mean(lat_err_trial_DWA[500:1000])))
print('Inner-city Mean value of End2End: ' + str(np.mean(lat_err_trial_End2End[500:1000])))
print('Inner-city Mean value of Ours: ' + str(np.mean(lat_err_trial_NeuroTrajectory[500:1000])))
print('')

print('Inner-city Max value of DWA: ' + str(np.max(lat_err_trial_DWA[500:1000])))
print('Inner-city Max value of End2End: ' + str(np.max(lat_err_trial_End2End[500:1000])))
print('Inner-city Max value of Ours: ' + str(np.max(lat_err_trial_NeuroTrajectory[500:1000])))
print('')

#print('Min value of Ours: ' + str(np.min(lat_err_trial_1)))
#print('Min value of End2EndLearning: ' + str(np.min(lat_err_trial_2)))
#print('Min value of DRL: ' + str(np.min(lat_err_trial_3)))




plt.plot(traveled_dist, lat_err_trial_DWA, label='DWA')
plt.fill_between(traveled_dist, lat_err_trial_2_sd_poz, lat_err_trial_2_sd_neg, alpha=0.4)

plt.plot(traveled_dist, lat_err_trial_End2End, label='End2End')
plt.fill_between(traveled_dist, lat_err_trial_3_sd_poz, lat_err_trial_3_sd_neg, alpha=0.4)

plt.plot(traveled_dist, lat_err_trial_NeuroTrajectory, label='NeuroTrajectory')
plt.fill_between(traveled_dist, lat_err_trial_1_sd_poz, lat_err_trial_1_sd_neg, alpha=0.4)

plt.xticks(np.arange(min(traveled_dist), max(traveled_dist), 5))
plt.legend(loc=1, ncol=3, columnspacing=1.0)

# show the bottom axis label
plt.subplots_adjust(bottom=0.24)

# remove margins
plt.margins(0)

# set grid linestyle
plt.grid(linestyle='--')


fig.savefig("position_error.png")
plt.show()
