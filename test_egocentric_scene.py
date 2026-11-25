import numpy as np

def generate_occupancy_map(scene, threshold, block_size=0.1):
    # 假设scene的shape是(B, 4096, 3)，其中B是样本数，4096是点的数目，每个点有(x, z, y)坐标
    
    # 将(x, z, y)坐标映射到(x, y)平面
    x_vals = scene[:, :, 0]  # 获取x坐标
    y_vals = scene[:, :, 2]  # 获取y坐标
    z_vals = scene[:, :, 1]  # 获取z坐标
    
    # 确定x和y的最小最大值，以便确定网格大小
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    
    # 计算网格的尺寸
    x_grid_size = int(np.ceil((x_max - x_min) / block_size))
    y_grid_size = int(np.ceil((y_max - y_min) / block_size))
    
    # 创建一个全为0的占用图
    occupancy_map = np.zeros((x_grid_size, y_grid_size))
    
    # 遍历场景中的每个点
    for i in range(scene.shape[0]):  # 遍历B个场景样本
        for j in range(scene.shape[1]):  # 遍历每个样本中的4096个点
            x = x_vals[i, j]
            y = y_vals[i, j]
            z = z_vals[i, j]
            
            # 将x, y坐标映射到对应的网格位置
            x_index = int((x - x_min) / block_size)
            y_index = int((y - y_min) / block_size)
            
            # 如果当前z值大于threshold，设置为占用（1）
            if z > threshold:
                occupancy_map[x_index, y_index] = 1
    
    return occupancy_map

# 示例使用
scene = np.random.rand(10, 4096, 3) * 10  # 示例场景数据
threshold = 0.5  # 假设阈值为0.5
occupancy_map = generate_occupancy_map(scene, threshold)

# 输出结果
import matplotlib.pyplot as plt
plt.imshow(occupancy_map, cmap='gray')
plt.savefig('occupancy_map.png')