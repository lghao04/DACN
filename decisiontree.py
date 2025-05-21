from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Tải dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Huấn luyện cây quyết định
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X, y)

# Hàm trích xuất nhánh
def print_tree_paths(tree, feature_names, target_names, node_id=0, path=None, depth=0):
    if path is None:
        path = []
    
    # Kiểm tra xem nút có phải là lá
    if tree.feature[node_id] == -2:  # Lá
        value = np.argmax(tree.value[node_id])
        print(f"{'  ' * depth}Leaf: Predict {target_names[value]} (Path: {' AND '.join(path)})")
    else:
        # Lấy đặc trưng và ngưỡng tại nút
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        
        # Nhánh nhỏ hơn hoặc bằng
        print_tree_paths(tree, feature_names, target_names, 
                         tree.children_left[node_id], path + [f"{feature} <= {threshold:.2f}"], depth + 1)
        # Nhánh lớn hơn
        print_tree_paths(tree, feature_names, target_names, 
                         tree.children_right[node_id], path + [f"{feature} > {threshold:.2f}"], depth + 1)

print("Decision Tree Paths:")
print_tree_paths(dt.tree_, feature_names, target_names)