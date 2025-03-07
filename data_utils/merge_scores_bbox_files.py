import pandas as pd
import argparse


def main(args):
    bbox_df = pd.read_csv(args.bbox_file)
    score_df = pd.read_csv(args.score_file)
    
    bbox_df.drop(columns=["finger"], inplace=True)
    score_df.drop(columns=["finger"], inplace=True)
    
    bbox_df.rename(columns={"Unnamed: 0": "joint_id"}, inplace=True)
    bbox_df["joint_id"] = bbox_df["joint_id"].apply(lambda x: x % 42)
    
    
    merged_df = pd.merge(
        score_df,
        bbox_df,
        left_on=['joint_id', 'patient_id', 'hand', 'joint'],
        right_on=['joint_id', 'patient_id', 'hand', 'joint'],
        how='inner'
    )
    
    merged_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bbox_file", type=str, required=True)
    parser.add_argument("--score_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
    