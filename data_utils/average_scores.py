import argparse
import pandas as pd

def transform_scores(input_file, output_file):
    """
    Transform the raw scores data into the desired format.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the transformed CSV file.
    """
    
    df = pd.read_csv(input_file)
    
    df.drop("expert_id", axis=1, inplace=True)
    df = df.rename(columns={df.columns[0]: "joint_id"})
    df["finger"] = df["finger"].fillna("none")
    df = df.dropna(subset=["score"])

    grouped = df.groupby(["joint_id", "patient_id", "hand", "joint", "finger", "disease"], dropna=False)["score"].mean().reset_index()
    grouped["score"] = grouped["score"].round()

    pivoted = grouped.pivot_table(
        index=["joint_id", "patient_id", "hand", "joint", "finger"],
        columns="disease",
        values="score",
        fill_value=0
    ).reset_index()

    pivoted.columns.name = None 
    pivoted = pivoted.rename(columns={"erosion": "erosion_score", "JSN": "jsn_score"})
    pivoted.sort_values(by=["patient_id", "joint_id"], ascending=[False, True], inplace=True)

    pivoted.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Transform raw scores data into the desired format.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the transformed CSV file.")
    args = parser.parse_args()

    transform_scores(args.input_csv, args.output_csv)