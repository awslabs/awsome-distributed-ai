import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read CSV
all_reduce_df = pd.read_csv('./all_reduce.csv')

# Convert duration to float
all_reduce_df[['duration_ms','unit']] = all_reduce_df['Duration'].str.split(' ',expand = True)
all_reduce_df['duration_ms']=all_reduce_df['duration_ms'].astype('float')

# Filter by Message Size
one_df_1GB = all_reduce_df.loc[all_reduce_df['Message_Size']=='1,073,741,824',]
one_df_2GB = all_reduce_df.loc[all_reduce_df['Message_Size']=='2,147,483,648',]

NCCL_Test_allreduce_1GB = 53.361
NCCL_Test_allreduce_2GB = 92.866

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.hist(one_df_1GB['duration_ms'], density=True, bins=100)  # density=False would make counts
ax1.set_ylabel('Probability')
ax1.set_xlabel('Duration ms')
ax1.set_xlim([20, 220])
ax1.axvline(NCCL_Test_allreduce_1GB, color='k', linestyle='dashed', linewidth=1)
ax1.set_title('All Reduce:Sum Message Size = 1GB')
ax1.legend(['NCCL Test Reported Time', 'NCCL Test Iterations'])

ax2.hist(one_df_2GB['duration_ms'], density=True, bins=100)  # density=False would make counts
ax2.set_ylabel('Probability')
ax2.set_xlabel('Duration ms')
ax2.set_xlim([20, 220])
ax2.axvline(NCCL_Test_allreduce_2GB, color='k', linestyle='dashed', linewidth=1)
ax2.set_title('All Reduce:Sum Message Size = 2GB')
ax2.legend(['NCCL Test Reported Time', 'NCCL Test Iterations'])

plt.tight_layout()

plt.savefig('all_reduce_sum.png')