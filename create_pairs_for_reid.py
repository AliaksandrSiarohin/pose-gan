import pandas as pd
from cmd import args
import pose_transform
import pose_utils
from itertools import permutations

args = args()


def filter_not_valid(df_keypoints):
    def check_valid(x):
        kp_array = pose_utils.load_pose_cords_from_strings(x['keypoints_y'], x['keypoints_x'])
        distractor = x['name'].startswith('-1') or x['name'].startswith('0000')
        return pose_transform.check_valid(kp_array) and not distractor
    return df_keypoints[df_keypoints.apply(check_valid, axis=1)].copy()


def make_pairs(df, pairs_for_each=10):
    fr, to = [], []
    for image_name in df['name']:
        fr_names = [image_name] * pairs_for_each
        to_names = df['name'].sample(n=pairs_for_each)
        fr += list(fr_names)
        to += list(to_names)
    pair_df = pd.DataFrame(index=range(len(fr)))
    pair_df['from'] = fr
    pair_df['to'] = to
    return pair_df


if __name__ == "__main__":
    df_keypoints = pd.read_csv(args.annotations_file_train, sep=':')
    df = filter_not_valid(df_keypoints)

    print ('Compute pair for train re-id...')
    pairs_df_train = make_pairs(df)
    print ('Number of pairs: %s' % len(pairs_df_train))
    pairs_df_train.to_csv('data/market-re-id-pairs.csv', index=False)
