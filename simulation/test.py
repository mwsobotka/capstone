import numpy as np
from parties import update_parties, merge_close_parties

def test_update_parties():
    # 5 voters near (0,0) who choose party 0
    # 5 voters near (4,0) who choose party 1
    voters = np.array([
        [-0.1,  0.0],
        [ 0.2, -0.1],
        [-0.2,  0.1],
        [ 0.1,  0.2],
        [ 0.0, -0.2],
        [ 3.9,  0.1],
        [ 4.1, -0.2],
        [ 4.2,  0.0],
        [ 3.8,  0.2],
        [ 4.0, -0.1],
    ])

    # 2 parties, far from their voters on purpose
    parties = np.array([
        [-2.0, 0.0],   # party 0 starts left of its supporters
        [ 6.0, 0.0],   # party 1 starts right of its supporters
    ])

    # first 5 voters choose party 0, next 5 voters choose party 1
    choices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    eta = 0.25  # move 25% of the way toward supporters

    print("=== test_update_parties ===")
    print("Original parties:\n", parties)

    new_parties = update_parties(voters, parties, choices, eta=eta)

    print("Updated parties:\n", new_parties)

    # manually compute supporter means to check
    mean_0 = voters[choices == 0].mean(axis=0)
    mean_1 = voters[choices == 1].mean(axis=0)
    print("Mean supporters for party 0:", mean_0)
    print("Mean supporters for party 1:", mean_1)

    # check one example numerically:
    expected_0 = parties[0] + eta * (mean_0 - parties[0])
    print("Expected updated party 0:", expected_0)
    print("Actual   updated party 0:", new_parties[0])

import numpy as np
from parties import update_parties, merge_close_parties

def test_update_parties():
    # 5 voters near (0,0) who choose party 0
    # 5 voters near (4,0) who choose party 1
    voters = np.array([
        [-0.1,  0.0],
        [ 0.2, -0.1],
        [-0.2,  0.1],
        [ 0.1,  0.2],
        [ 0.0, -0.2],
        [ 3.9,  0.1],
        [ 4.1, -0.2],
        [ 4.2,  0.0],
        [ 3.8,  0.2],
        [ 4.0, -0.1],
    ])

    # 2 parties, far from their voters on purpose
    parties = np.array([
        [-2.0, 0.0],   # party 0 starts left of its supporters
        [ 6.0, 0.0],   # party 1 starts right of its supporters
    ])

    # first 5 voters choose party 0, next 5 voters choose party 1
    choices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    eta = 0.25  # move 25% of the way toward supporters

    print("=== test_update_parties ===")
    print("Original parties:\n", parties)

    new_parties = update_parties(voters, parties, choices, eta=eta)

    print("Updated parties:\n", new_parties)

    # manually compute supporter means to check
    mean_0 = voters[choices == 0].mean(axis=0)
    mean_1 = voters[choices == 1].mean(axis=0)
    print("Mean supporters for party 0:", mean_0)
    print("Mean supporters for party 1:", mean_1)

    # check one example numerically:
    expected_0 = parties[0] + eta * (mean_0 - parties[0])
    print("Expected updated party 0:", expected_0)
    print("Actual   updated party 0:", new_parties[0])

def test_merge_close_parties():
    # three parties:
    #  - party 0 at (0,0)
    #  - party 1 very close to party 0
    #  - party 2 far away
    parties = np.array([
        [0.0, 0.0],   # p0
        [0.2, 0.1],   # p1 (close to p0)
        [3.0, 3.0],   # p2 (far)
    ])

    print("\n=== test_merge_close_parties ===")
    print("Original parties:\n", parties)

    # set merge distance threshold
    d_merge = 0.5

    new_parties = merge_close_parties(parties, d_merge=d_merge)

    print("Merged parties (d_merge=0.5):\n", new_parties)
    print("Number of parties before:", parties.shape[0])
    print("Number of parties after: ", new_parties.shape[0])

if __name__ == "__main__":
    test_update_parties()
    test_merge_close_parties()


