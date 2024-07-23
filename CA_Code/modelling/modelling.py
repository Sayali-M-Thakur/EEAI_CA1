from model.randomforest import RandomForest
from Config import Config


def model_predict(data, df, name):
    print("Chained RandomForest")

    models = []
    for i, target_col in enumerate(Config.TYPE_COLS):
        if not data.y[i]:
            continue  # Skip if no data for this target variable
        model = RandomForest(f"RandomForest_{target_col}", data.X, data.y[i])
        model.train(data)
        models.append(model)

    if not models:
        print("No models trained due to lack of data. Skipping...")
        return

    # Make predictions sequentially for each model
    predictions = []
    for model in models:
        model.predict(data.X_test)
        predictions.append(model.predictions)

    # Print results for each target variable
    for i, target_col in enumerate(Config.TYPE_COLS):
        if not data.y[i]:
            continue
        print(f"\nResults for {target_col}")
        model_evaluate(models[i], data)


def model_evaluate(model, data):
    model.print_results(data)