"""#setp1 
定义空列表 total_tokens = []
遍历训练集：
    执行sample_label_tokens ,以得到经处理后的tokens,
    将tokens添加到total_tokens中
聚类,结果进行保存，作为step2的一个输入，用于增强特征
#step2
继续遍历训练集，训练
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import time

# Step 1: Generate 10000 random 3x4x4 matrices (simulate image patches)
num_matrices = 200000
matrix_shape = (3, 16, 16)

# Generate random 3x4x4 matrices (each matrix has 48 elements after flattening)
data = np.random.rand(num_matrices, np.prod(matrix_shape))  # shape: (10000, 48)

# Step 2: Apply Mini-Batch KMeans clustering with parallel computation
num_clusters = 80

# Record start time
start_time = time.time()

# Perform Mini-Batch KMeans clustering with parallel computation  MiniBatchKMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)

# Record end time
end_time = time.time()

# Step 3: Output the time taken
execution_time = end_time - start_time
print(f"Mini-Batch K-means clustering completed in 16*16{execution_time:.4f} seconds.")






