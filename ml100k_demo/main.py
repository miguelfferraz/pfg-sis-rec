import argparse
from collections import defaultdict

import pandas as pd
from surprise import Dataset, KNNBaseline, get_dataset_dir


def get_top_recommendation(predictions, n=3):
    res = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        res[uid].append((iid, est))

    for uid, user_ratings in res.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        res[uid] = user_ratings[:n]

    return res


def read_item_names():
    file_name = get_dataset_dir() + "/ml-100k/ml-100k/u.item"
    rid_to_name = {}
    with open(file_name, encoding="ISO-8859-1") as f:
        for line in f:
            line = line.split("|")
            rid_to_name[line[0]] = line[1]

    return rid_to_name


def train(dataset: Dataset):
    train_set = dataset.build_full_trainset()

    knn = KNNBaseline(sim_option={"name": "cosine", "user_based": False})
    knn.fit(train_set)

    test_set = train_set.build_anti_testset()
    predictions = knn.test(test_set)

    return predictions


def export_to_csv(top_recommendations, rid_to_name, filename="recommendations.csv"):
    data = []
    for uid, user_ratings in top_recommendations.items():
        for _, (iid, score) in enumerate(user_ratings, 1):
            data.append(
                {
                    "user_id": uid,
                    "movie_id": iid,
                    "movie_name": rid_to_name.get(iid, "Unknown"),
                    "predicted_rating": round(score, 3),
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Recommendations exported to '{filename}' with {len(data)} records.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-csv", metavar="FILENAME", default=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    dataset = Dataset.load_builtin("ml-100k")
    predictions = train(dataset)

    top_recommendations = get_top_recommendation(predictions, n=args.top_n)
    rid_to_name = read_item_names()

    if args.export_csv:
        export_to_csv(top_recommendations, rid_to_name, args.export_csv)
    else:
        print("\nTop recommendations per user (showing first 10 users):")
        for uid, user_ratings in list(top_recommendations.items())[:10]:
            movies = [rid_to_name[iid] for (iid, _) in user_ratings]
            print(f"User {uid}: {movies}")


if __name__ == "__main__":
    main()
