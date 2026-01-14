from transaction.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    print("Training started...")

    pipeline = TrainingPipeline(data_path="transactions.csv")
    pipeline.run()

    print("Training completed successfully!")
