from utils.data_loader import load_subject_timeframe

def main(data_file: str, output_file: str):
    # Load timeframe from the data file
    features, labels = load_subject_timeframe(data_file, drop_non_study=False, imputer_strategy='mean', shorten_non_study=False)
    
    # Save the features and labels to a CSV file
    features.assign(labels=labels).to_csv(output_file, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from subject data.")
    parser.add_argument("--data_file", type=str, help="Path to the input data file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file to save features and labels.")
    args = parser.parse_args()

    main(args.data_file, args.output_file)