from hypothesis import given, strategies as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Tải và huấn luyện k-NN
iris = load_iris()
X, y = iris.data, iris.target
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

@given(
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(0.0, 7.0),
    petal_width=st.floats(0.0, 0.8)  # x[3] <= 0.8
)
def test_setosa_branch(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [0], f"Expected setosa (0), got {predicted}"


@given(
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(0.0, 4.95),
    petal_width=st.floats(0.81, 1.65)  # x[3] > 0.8 and <= 1.65
)
def test_versicolor_branch_1(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [1], f"Expected versicolor (1), got {predicted}"


@given(
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(0.0, 4.95),
    petal_width=st.floats(0.81, 1.65)  # x[3] > 0.8 and <= 1.65
)
def test_versicolor_branch_1(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [1], f"Expected versicolor (1), got {predicted}"

@given(
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(4.96, 7.0),
    petal_width=st.floats(0.81, 1.55)
)
def test_virginica_branch_2(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [2], f"Expected virginica (2), got {predicted}"

@given(
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(4.96, 7.0),
    petal_width=st.floats(1.56, 1.75)
)
def test_versicolor_branch_2(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [1], f"Expected versicolor (1), got {predicted}"

@given(
    sepal_length=st.floats(4.0, 5.95),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(0.0, 4.85),
    petal_width=st.floats(1.76, 2.5)
)
def test_versicolor_branch_3(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [1], f"Expected versicolor (1), got {predicted}"

@given(
    sepal_length=st.floats(5.96, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(0.0, 4.85),
    petal_width=st.floats(1.76, 2.5)
)
def test_virginica_branch_3(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [2], f"Expected virginica (2), got {predicted}"

@given(
    sepal_length=st.floats(4.0, 8.0),
    sepal_width=st.floats(2.0, 4.5),
    petal_length=st.floats(4.86, 7.0),
    petal_width=st.floats(1.76, 2.5)
)
def test_virginica_branch_4(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted = knn.predict(sample)
    assert predicted == [2], f"Expected virginica (2), got {predicted}"

# Chạy thử
if __name__ == "__main__":
    import pytest
    pytest.main(["-v", "--pyargs", __file__])