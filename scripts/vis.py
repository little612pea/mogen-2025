import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载数据
gt_data = np.load("../motion_embedding_ground truth.npy")  # shape (3072, 256)
vald_data = np.load("../motion_embedding_vald.npy")  # shape (1024, 256)

# 创建标签以区分两个数据集
labels = np.array([0] * gt_data.shape[0] + [1] * vald_data.shape[0])  # 0 表示 gt，1 表示 vald

# 合并数据
combined_data = np.vstack((gt_data, vald_data))  # shape (4096, 256)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(combined_data)  # shape (4096, 2)

print("t-SNE results shape:", tsne_results.shape)

# 分离 t-SNE 结果
gt_tsne = tsne_results[:gt_data.shape[0]]  # gt 数据的 t-SNE 结果
vald_tsne = tsne_results[gt_data.shape[0]:]  # vald 数据的 t-SNE 结果

# 可视化
plt.figure(figsize=(8, 8))

# 绘制 gt 数据分布
plt.scatter(
    gt_tsne[:, 0],
    gt_tsne[:, 1],
    c='blue',
    label='Ground Truth',
    alpha=0.6,
    s=10
)

# 绘制 vald 数据分布
plt.scatter(
    vald_tsne[:, 0],
    vald_tsne[:, 1],
    c='red',
    label='Validation',
    alpha=0.6,
    s=10
)

# 添加图例和标题
plt.legend()
plt.title('t-SNE Visualization of GT and Validation Data')

# 显示图片
plt.show()

# 保存图片
plt.savefig('tSNE_gt_vald_mixamo_motion-x.png')