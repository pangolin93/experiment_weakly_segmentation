from weakseg import RANDOM_STATE
from sklearn.model_selection import train_test_split

num_tot = 33

num_train = 3
num_weak = 23
num_test = 7

import logging
logger = logging.getLogger(__name__)

def get_train_weak_val_indexes(num_tot = 33, num_train = 3, num_weak = 23, num_test = 7):

    indexes_train_weak, indexes_val = train_test_split(
        range(num_tot), 
        test_size=num_test/num_tot, 
        shuffle=True,
        random_state=RANDOM_STATE)

    indexes_train, indexes_weak = train_test_split(
        indexes_train_weak, 
        train_size=num_train/(len(indexes_train_weak)), 
        shuffle=True,
        random_state=RANDOM_STATE)

    return indexes_train, indexes_weak, indexes_val


if __name__ == '__main__':

    num_tot = 33

    num_train = 3
    num_weak = 23
    num_test = 7

    indexes_train, indexes_weak, indexes_val = get_train_weak_val_indexes(num_tot, num_train, num_weak, num_test)

    print((indexes_train, indexes_weak, indexes_val))
