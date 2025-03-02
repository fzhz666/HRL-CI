import pickle
import numpy as np

data_total = {"final_state": np.zeros(1000),
              "time": np.zeros(1000),
              "robot_path": [],
              "target_path": []}
for i in range(5):
    # FILE_NAME = 'pH-DDPG-' + str(i+1) + '_0_199_simple.p'
    FILE_NAME = 'pH-DDPG-' + str(i+1) + '_0_199.p'
    run_data = pickle.load(open('../record_data/' + FILE_NAME, 'rb'))
    data_total["final_state"][i*200: (i+1)*200] = run_data["final_state"]
    data_total["time"][i*200: (i+1)*200] = run_data["time"]
    data_total["robot_path"] += run_data["robot_path"]
    data_total["target_path"] += run_data["target_path"]

# pickle.dump(data_total, open('../record_data/' + 'pH-DDPG-1-5_0_999_simple.p', 'wb+'))
pickle.dump(data_total, open('../record_data/' + 'pH-DDPG-1-5_0_999.p', 'wb+'))


