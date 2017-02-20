# detect author gender from essay corpus

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import feature_extractor as fe
import numpy
import re
import sys


def main():
    """ main function, called if module is run as main program"""

    if len(sys.argv) == 1:
        print('usage: bawe_gender_classifier.py <CORPUS_DIR> <CONF_FILE_NAME>')
        exit(0)
    data_dir = sys.argv[1]

    # read file list into dictionary mapping file id to gender
    with open(data_dir + '/BAWE_balanced_subset.csv', 'r') as gender_file:
        meta_lines = [line.rstrip().split(',') for line in gender_file]
        gender_dict = {row[0]: row[1] for row in meta_lines[1:]}

    # read essay contents and gender labels into lists
    essays = []
    gender_labels = []
    for student, gender in gender_dict.items():
        with open('%s/%s.txt' % (data_dir, student)) as f:
            text = f.read()
            # remove vestigial xml
            text = re.sub('<[^<]+?>', '', text)
            essays.append(text)
            gender_labels.append(gender)

    # read conf file
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as conf_file:
            conf_all = set(line.strip() for line in conf_file)
    else:
        conf_all = []

    # compute score, for each line in the config individually and all together
    # note: preprocessing every time is very wasteful but i did not want to
    #       change the feature_extractor interfaces from what was given
    # each line individually
    confs = [line for line in conf_all]
    if len(conf_all) > 1:
        # all lines together
        confs.append([line for line in conf_all])
    for conf in confs:
        print('computing score for: %s... '
              % (conf if conf else 'all features'), end='')
        features = fe.extract_features(essays, conf)
        print(cross_val_score(GaussianNB(), features, gender_labels,
                              scoring='accuracy', cv=10).mean())


if __name__ == "__main__":
    main()
