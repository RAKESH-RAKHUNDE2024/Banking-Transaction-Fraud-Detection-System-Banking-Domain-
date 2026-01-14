from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from transaction.logger import logger
from transaction.utils import save_object
from transaction.config import MODEL_PATH

class ModelTrainer:
    def train(self, preprocessor, X_train, y_train, X_test, y_test):
        logger.info("Training model started...")

        model = LogisticRegression(max_iter=1000, class_weight="balanced")

        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        logger.info("Model training completed.")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
        logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)}")

        save_object(MODEL_PATH, clf)

        logger.info(f"Model saved at: {MODEL_PATH}")

        return clf