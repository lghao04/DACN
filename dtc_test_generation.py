from hypothesis import given, strategies as st
from hypothesis import settings, HealthCheck
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from hypothesis import assume
# Tải và huấn luyện k-NN
iris = load_iris()
X, y = iris.data, iris.target
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

settings.load_profile("colab")

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(max_value=2.45),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_width=st.floats(0.0, 2.5),
)
def test_setosa(petal_length, sepal_length, sepal_width, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 0, f"Expected setosa (0), got {predicted}"


# 2. Leaf: Predict versicolor (petal length > 2.45 AND petal width <= 1.75 AND petal length <= 4.95 AND petal width <= 1.65)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=2.45, max_value=4.95),
    petal_width=st.floats(max_value=1.65),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
)
def test_versicolor_branch_1(petal_length, petal_width, sepal_length, sepal_width):
    assume(petal_width <= 1.75)
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 1, f"Expected versicolor (1), got {predicted}"


# 3. Leaf: Predict virginica (petal length > 2.45 AND petal width <= 1.75 AND petal length <= 4.95 AND petal width > 1.65)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=2.45, max_value=4.95),
    petal_width=st.floats(min_value=1.66, max_value=1.75),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
)
def test_virginica_branch_1(petal_length, petal_width, sepal_length, sepal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 2, f"Expected virginica (2), got {predicted}"


# 4. Leaf: Predict virginica (petal length > 2.45 AND petal width <= 1.75 AND petal length > 4.95 AND petal width <= 1.55)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=4.95, max_value=7.0),
    petal_width=st.floats(max_value=1.55),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
)
def test_virginica_branch_2(petal_length, petal_width, sepal_length, sepal_width):
    assume(petal_width <= 1.75)
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 2, f"Expected virginica (2), got {predicted}"


# 5. Leaf: Predict versicolor (petal length > 2.45 AND petal width <= 1.75 AND petal length > 4.95 AND petal width > 1.55 AND petal length <= 5.45)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=4.95, max_value=5.45),
    petal_width=st.floats(min_value=1.56, max_value=1.75),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
)
def test_versicolor_branch_2(petal_length, petal_width, sepal_length, sepal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 1, f"Expected versicolor (1), got {predicted}"


# 6. Leaf: Predict virginica (petal length > 2.45 AND petal width <= 1.75 AND petal length > 4.95 AND petal width > 1.55 AND petal length > 5.45)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=5.45, max_value=7.0),
    petal_width=st.floats(min_value=1.56, max_value=1.75),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
)
def test_virginica_branch_3(petal_length, petal_width, sepal_length, sepal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 2, f"Expected virginica (2), got {predicted}"


# 7. Leaf: Predict versicolor (petal length > 2.45 AND petal width > 1.75 AND petal length <= 4.85 AND sepal length <= 5.95)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=2.45, max_value=4.85),
    petal_width=st.floats(min_value=1.76),
    sepal_length=st.floats(max_value=5.95),
    sepal_width=st.floats(2.0, 4.5),
)
def test_versicolor_branch_3(petal_length, petal_width, sepal_length, sepal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 1, f"Expected versicolor (1), got {predicted}"


# 8. Leaf: Predict virginica (petal length > 2.45 AND petal width > 1.75 AND petal length <= 4.85 AND sepal length > 5.95)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=2.45, max_value=4.85),
    petal_width=st.floats(min_value=1.76),
    sepal_length=st.floats(min_value=5.95),
    sepal_width=st.floats(2.0, 4.5),
)
def test_virginica_branch_4(petal_length, petal_width, sepal_length, sepal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 2, f"Expected virginica (2), got {predicted}"


# 9. Leaf: Predict virginica (petal length > 2.45 AND petal width > 1.75 AND petal length > 4.85)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(
    petal_length=st.floats(min_value=4.85, max_value=7.0),
    petal_width=st.floats(min_value=1.76),
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
)
def test_virginica_branch_5(petal_length, petal_width, sepal_length, sepal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)[0]
    assert predicted == 2, f"Expected virginica (2), got {predicted}"
# ✅ Chạy thử tất cả các test một lần mỗi cái
# print("Running hypothesis tests (1 example each)...")
# test_setosa_branch()
# test_versicolor_branch_1()
# test_virginica_branch_2()
# test_versicolor_branch_2()
# test_versicolor_branch_3()
# test_virginica_branch_3()
# test_virginica_branch_4()
# print("✅ All tests passed (for at least 1 example each)")

print("🧪 Running hypothesis tests with multiple examples...")
for test_func in [test_setosa_branch, test_versicolor_branch_1, test_virginica_branch_2]:
    try:
        test_func()  # sẽ chạy 30 ví dụ ngẫu nhiên theo cấu hình ở trên
        print(f"✅ {test_func.__name__} PASSED")
    except AssertionError as e:
        print(f"❌ {test_func.__name__} FAILED: {e}")