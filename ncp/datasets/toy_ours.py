import numpy as np
from ncp import tools


def generate_toy_ours_dataset():
    random = np.random.RandomState(0)

    def data_mean(x):
        return x * np.sin(x)

    def randn_like(x):
        return np.random.randn(*x.shape)

    training_range = [0, 10]
    testing_range = [0, 10]
    # Ours was
    # testing_range = [-0.5, 10.5]
    N_train = 3
    N_test = 100
    x_train = np.random.uniform(
        low=training_range[0], high=training_range[1], size=N_train
    )
    eps1, eps2 = randn_like(x_train), randn_like(x_train)
    y_train = data_mean(x_train) + 0.3 * eps1 + 0.3 * x_train * eps2
    x_test = np.random.uniform(low=testing_range[0], high=testing_range[1], size=N_test)
    y_test = data_mean(x_test)

    x_mu, x_sigma = x_train.mean(), x_train.std()
    y_mu, y_sigma = y_train.mean(), y_train.std()
    x_test, y_test = (x_test - x_mu) / x_sigma, (y_test - y_mu) / y_sigma
    x_train, y_train = (x_train - x_mu) / x_sigma, (y_train - y_mu) / y_sigma

    training_range = [x_train.min(), x_train.max()]

    train = tools.AttrDict(inputs=x_train[:, None], targets=y_train[:, None])
    test = tools.AttrDict(inputs=x_test[:, None], targets=y_test[:, None])
    return tools.AttrDict(
        domain=np.array(training_range)[:, None], train=train, test=test, target_scale=1
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.neural_network import MLPRegressor

    dataset = generate_toy_ours_dataset()

    regr = MLPRegressor(hidden_layer_sizes=(50, 50), random_state=1, max_iter=500).fit(
        dataset.train.inputs, dataset.train.targets
    )
    pred = regr.predict(dataset.test.inputs)

    _, ax = plt.subplots()
    ax.plot(dataset.test.inputs, dataset.test.targets, "o", label="test")
    ax.plot(dataset.test.inputs, pred, "o", color="pink", label="pred")
    ax.plot(dataset.train.inputs, dataset.train.targets, "o", label="train")
    ax.legend()

    plt.show()
