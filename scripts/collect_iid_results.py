import os
import pandas as pd


ROOT = "experiments/iid_main_table"

OUTPUT = "iid_main_table_results.csv"


def main():

    records = []

    for dataset_dir in sorted(os.listdir(ROOT)):

        if not dataset_dir.startswith("iid_"):
            continue

        dataset = dataset_dir.replace("iid_", "")

        dataset_path = os.path.join(ROOT, dataset_dir)

        if not os.path.isdir(dataset_path):
            continue


        for seed_dir in sorted(os.listdir(dataset_path)):

            if not seed_dir.startswith("seed"):
                continue

            seed = seed_dir.replace("seed", "")

            csv_path = os.path.join(
                dataset_path,
                seed_dir,
                "results.csv"
            )

            if not os.path.exists(csv_path):
                print("Missing:", csv_path)
                continue


            try:
                df = pd.read_csv(csv_path)

                # 添加来源信息
                df.insert(0, "dataset", dataset)
                df.insert(1, "seed", seed)

                records.append(df)

                print(
                    "Loaded:",
                    dataset,
                    seed,
                    df.shape
                )

            except Exception as e:
                print(
                    "Failed:",
                    csv_path,
                    e
                )


    if len(records) == 0:
        print("No results found!")
        return


    all_df = pd.concat(
        records,
        ignore_index=True
    )


    all_df.to_csv(
        OUTPUT,
        index=False
    )


    print("\nSaved:")
    print(OUTPUT)

    print("\nShape:")
    print(all_df.shape)

    print("\nColumns:")
    print(all_df.columns.tolist())


if __name__ == "__main__":
    main()
