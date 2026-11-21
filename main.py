from datasets import load_dataset, concatenate_datasets

if __name__ == '__main__':

    # Load LIAR2
    dataset = load_dataset("chengxuphd/liar2")

    # Combine the original splits
    combined = concatenate_datasets([
        dataset["train"],
        dataset["validation"],
        dataset["test"]
    ])

    # Convert to a pandas DataFrame
    df = combined.to_pandas()

    # Keep only the four labels used in our project
    df = df[df["label"].isin([0, 1, 4, 5])].copy()

    # Convert to binary classification:
    # 0 = False, 1 = True
    df["binary_label"] = df["label"].map({
        0: 0,  # Pants on Fire
        1: 0,  # False
        4: 1,  # Mostly True
        5: 1  # True
    })

    # Retain only the columns needed for our experiments
    df = df[["statement", "speaker", "binary_label"]]

    print("Total statements:", len(df))
    print("\nClass distribution:")
    print(df["binary_label"].value_counts())

    print("\nUnique speakers:", df["speaker"].nunique())
    print("Missing speakers:", df["speaker"].isna().sum())
    print("\nFirst five records:")
    print(df.head())

    from sklearn.model_selection import (
        train_test_split,
        GroupShuffleSplit
    )

    # Random 80/20 split
    random_train, random_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["binary_label"]
    )

    # Speaker-independent split
    group_split = GroupShuffleSplit(
        n_splits=200,
        test_size=0.2,
        random_state=42
    )

    target_test_fraction = 0.2
    target_positive_fraction = df["binary_label"].mean()

    best_score = float("inf")
    best_indices = None

    for train_idx, test_idx in group_split.split(
            df,
            y=df["binary_label"],
            groups=df["speaker"]
    ):
        test_fraction = len(test_idx) / len(df)
        positive_fraction = df.iloc[test_idx]["binary_label"].mean()

        score = (
                abs(test_fraction - target_test_fraction)
                + abs(positive_fraction - target_positive_fraction)
        )

        if score < best_score:
            best_score = score
            best_indices = (train_idx, test_idx)

    train_idx, test_idx = best_indices

    speaker_train = df.iloc[train_idx]
    speaker_test = df.iloc[test_idx]


    # Print statistics for each split
    def print_split_info(name, train, test):
        print(f"\n{name}")
        print("Training statements:", len(train))
        print("Testing statements:", len(test))

        print("Training class distribution:")
        print(train["binary_label"].value_counts(normalize=True))

        print("Testing class distribution:")
        print(test["binary_label"].value_counts(normalize=True))

        train_speakers = set(train["speaker"])
        test_speakers = set(test["speaker"])

        overlap = train_speakers.intersection(test_speakers)
        print("Overlapping speakers:", len(overlap))


    print_split_info("Random Split", random_train, random_test)
    print_split_info(
        "Speaker-Independent Split",
        speaker_train,
        speaker_test
    )

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    results = []


    def evaluate_model(name, model, train, test):
        # Extract statements and labels
        X_train = train["statement"]
        y_train = train["binary_label"]

        X_test = test["statement"]
        y_test = test["binary_label"]

        # Train and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro",
            zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro",
            zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro",
            zero_division=0
        )

        # Store results for CSV export
        results.append({
            "Experiment": name,
            "Training Samples": len(train),
            "Testing Samples": len(test),
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1": f1
        })

        print(f"\n{name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Macro-F1:  {f1:.4f}")


    def create_lr_model():
        return Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])


    # Random split
    evaluate_model(
        "Random Split - Logistic Regression + TF-IDF",
        create_lr_model(),
        random_train,
        random_test
    )

    evaluate_model(
        "Random Split - Majority Baseline",
        DummyClassifier(strategy="most_frequent"),
        random_train,
        random_test
    )

    # Speaker-independent split
    evaluate_model(
        "Speaker-Independent - Logistic Regression + TF-IDF",
        create_lr_model(),
        speaker_train,
        speaker_test
    )

    evaluate_model(
        "Speaker-Independent - Majority Baseline",
        DummyClassifier(strategy="most_frequent"),
        speaker_train,
        speaker_test
    )

    from sklearn.svm import LinearSVC


    def create_svm_model():
        return Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LinearSVC(
                random_state=42
            ))
        ])


    evaluate_model(
        "Random Split - Linear SVM + TF-IDF",
        create_svm_model(),
        random_train,
        random_test
    )

    evaluate_model(
        "Speaker-Independent - Linear SVM + TF-IDF",
        create_svm_model(),
        speaker_train,
        speaker_test
    )

    import re
    import numpy as np

    from collections import Counter
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    FUNCTION_WORDS = [
        "the", "and", "but", "because", "of",
        "to", "in", "is", "for", "a"
    ]


    def extract_stylometric_features(statements):
        features = []

        for statement in statements:
            text = str(statement).lower()

            # Extract words
            words = re.findall(r"\b[a-z]+\b", text)
            word_count = len(words)
            safe_count = max(word_count, 1)

            # Statement length and average word length
            avg_word_length = (
                    sum(len(word) for word in words) / safe_count
            )

            row = [
                word_count,
                avg_word_length
            ]

            # Punctuation frequencies
            for punctuation in [",", ".", "?", "!"]:
                row.append(text.count(punctuation) / safe_count)

            # Function-word frequencies
            counts = Counter(words)

            for word in FUNCTION_WORDS:
                row.append(counts[word] / safe_count)

            features.append(row)

        return np.array(features, dtype=float)


    def create_stylometric_lr():
        return Pipeline([
            ("features", FunctionTransformer(
                extract_stylometric_features
            )),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])


    def create_stylometric_svm():
        return Pipeline([
            ("features", FunctionTransformer(
                extract_stylometric_features
            )),
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                random_state=42
            ))
        ])


    # Logistic Regression with stylometry
    evaluate_model(
        "Random Split - LR + Stylometry",
        create_stylometric_lr(),
        random_train,
        random_test
    )

    evaluate_model(
        "Speaker-Independent - LR + Stylometry",
        create_stylometric_lr(),
        speaker_train,
        speaker_test
    )

    # Linear SVM with stylometry
    evaluate_model(
        "Random Split - SVM + Stylometry",
        create_stylometric_svm(),
        random_train,
        random_test
    )

    evaluate_model(
        "Speaker-Independent - SVM + Stylometry",
        create_stylometric_svm(),
        speaker_train,
        speaker_test
    )

    from sklearn.pipeline import FeatureUnion


    def create_combined_model(classifier):
        return Pipeline([
            ("features", FeatureUnion([
                ("tfidf", TfidfVectorizer()),
                ("stylometry", Pipeline([
                    ("extract", FunctionTransformer(
                        extract_stylometric_features,
                        validate=False
                    )),
                    ("scaler", StandardScaler())
                ]))
            ], transformer_weights={
                "tfidf": 1.0,
                "stylometry": 0.25
            })),
            ("classifier", classifier)
        ])


    def create_combined_lr():
        return create_combined_model(
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )


    def create_combined_svm():
        return create_combined_model(
            LinearSVC(
                random_state=42,
                max_iter=10000
            )
        )


    # Logistic Regression with combined features
    evaluate_model(
        "Random Split - LR + Combined",
        create_combined_lr(),
        random_train,
        random_test
    )

    evaluate_model(
        "Speaker-Independent - LR + Combined",
        create_combined_lr(),
        speaker_train,
        speaker_test
    )

    # Linear SVM with combined features
    evaluate_model(
        "Random Split - SVM + Combined",
        create_combined_svm(),
        random_train,
        random_test
    )

    evaluate_model(
        "Speaker-Independent - SVM + Combined",
        create_combined_svm(),
        speaker_train,
        speaker_test
    )

    from pathlib import Path
    import pandas as pd

    # Create the results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Export experimental results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        results_dir / "metrics.csv",
        index=False
    )

    print("\nResults saved to results/metrics.csv")






