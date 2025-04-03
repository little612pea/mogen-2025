import numpy as np

def create_motion_tuple(motion_array):
    # 确保 motion_array 是一个 ndarray
    if isinstance(motion_array, np.ndarray):
        # 创建包含motion数据的字典
        motion_tuple = {
            'motion': motion_array,  # 已传入的 ndarray
            'text': [
                'the man buries his head in his arms and cry in despair, and finally crouch down on ones knees',
                'the man buries his head in his arms and cry in despair, and finally crouch down on ones knees',
                'the man buries his head in his arms and cry in despair, and finally crouch down on ones knees'
            ],
            'lengths': np.array([120, 120, 120]),  # 长度数组
            'num_samples': 1,  # 样本数
            'num_repetitions': 3  # 重复次数
        }
        return motion_tuple
    else:
        raise ValueError("输入的 motion_array 必须是一个 ndarray。")

# 示例：使用一个随机生成的 motion 数组
motion_array = np.load("results.npy",allow_pickle=True)

# 创建元组
motion_data = create_motion_tuple(motion_array)

# 打印结果
print(motion_data)
output_path = 'motion_data.npy'  # 你可以修改为你希望保存的路径
np.save(output_path, motion_data, allow_pickle=True)

print(f"motion_data 已保存为 {output_path}")