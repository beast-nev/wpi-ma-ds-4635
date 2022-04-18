import numpy as np

time_interval = 60

def open_csv():
    # read CSV file
    training_data_np = np.genfromtxt('data/train.csv', delimiter=',')
    return training_data_np[1:]

def condense_data(train):
    data = np.empty((1, 15))
    for sequence in range(int(len(train)/time_interval)):
        start_point = sequence * time_interval
        current_range = train[start_point:start_point + time_interval]
        means = np.mean(current_range[:, 3:], axis=0)
        row = np.concatenate((current_range[0, 0:2], means))
        row = np.reshape(row, (1, 15))
        if sequence == 0:
            data = np.reshape(row, (1, 15))
        else:
            data = np.append(data, row, axis=0)

    print(data)
    return data

if __name__ == "__main__":
    training_data = open_csv()
    condense_data(training_data)