import numpy as np
from matplotlib import pyplot as plt

version = 'sc_v64/'

train_mean_info = np.load(version +'/train_info.npy')
test_mean_info = np.load(version +'/test_info.npy')



print('train_mean_info',train_mean_info)
print('test_mean_info',test_mean_info)

test_mae_list = test_mean_info[:,-4]
test_f_list = test_mean_info[:,-1]

train_mae_list = train_mean_info[:,-4]

min_index = np.argmin(test_mae_list)

print('test_mae_list',test_mae_list)
print('min_index,min_value',min_index, test_mae_list[min_index],test_f_list[min_index])

plt.plot(test_mae_list)
#plt.plot(train_mae_list)

plt.show()



