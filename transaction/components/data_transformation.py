import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from transaction.logger import logger

class DataTransformation:
    def __init__(self):
        self.target_column = "fraud_flag"

    def get_preprocessor(self, df: pd.DataFrame):
        logger.info("Creating preprocessing pipeline...")

        drop_columns = ["transaction_id", "timestamp"]  # timestamp can be used later, but drop for now

        X = df.drop(columns=[self.target_column], errors="ignore")
        X = X.drop(columns=drop_columns, errors="ignore")

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numeric columns: {numeric_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("scaler", StandardScaler())
                ]), numeric_cols),

                ("cat", Pipeline(steps=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical_cols)
            ]
        )

        return preprocessor, drop_columns

    def transform_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        logger.info("Transforming train and test data...")

        preprocessor, drop_columns = self.get_preprocessor(train_df)

        X_train = train_df.drop(columns=[self.target_column], errors="ignore").drop(columns=drop_columns, errors="ignore")
        y_train = train_df[self.target_column]

        X_test = test_df.drop(columns=[self.target_column], errors="ignore").drop(columns=drop_columns, errors="ignore")
        y_test = test_df[self.target_column]

        return preprocessor, X_train, y_train, X_test, y_test