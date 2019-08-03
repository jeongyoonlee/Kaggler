import numpy as np
import profile
from kaggler.online_model import ClassificationTree
from tqdm import tqdm


N_OBS = 10000
N_FEATURE = 20


def test():
    data = np.random.randint(0, 1000, size=(N_OBS, N_FEATURE))
    y = np.random.randint(2, size=N_OBS)

    train = data[0:50000]
    ytrain = y[0:50000]
    test = data[50000:]
    ytest = y[50000:]

    learner = ClassificationTree(number_of_features=N_FEATURE)

    for t, x in enumerate(tqdm(train)):
        learner.update(x, ytrain[t])

    correct_num = 0
    for t, x in enumerate(tqdm(test)):
        y_pred = learner.predict(x)
        if y_pred == ytest[t]:
            correct_num += 1

    print(correct_num)


if __name__ == '__main__':
    profile.run("test()")
