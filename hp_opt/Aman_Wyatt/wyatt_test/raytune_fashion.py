from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.runtime_context import get_runtime_context
from ray.tune.search.bayesopt import BayesOptSearch # required for bayesian optimization
from ray.tune.search.optuna import OptunaSearch # required for optuna optimization
from fashion import TrainConfig, train_eval, set_seed

def ray_train(config):
    ctx = get_runtime_context()
    # convert ray tune config dict into your dataclass
    cfg = TrainConfig(
        model_type="cnn", #config["model_type"],
        batch_size=512, #config["batch_size"],
        epochs=15, # short for tuning, can increase later
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        dropout=config["dropout"],
        seed=42,
        val_split=0.1,
        num_workers=4,
        data_dir="./data",
        save_path="./ray_tune_model.pt"
    )

    set_seed(cfg.seed)
    # Run existing train_eval, but instead of printing only, report metrics back to Ray Tune
    # report metrics to Ray Tune
    set_results = train_eval(cfg) 

    # report validation accuracy
    # tune.report(val_acc=set_results["val_acc"], test_acc=set_results["test_acc"])
    # tune.report({"val_acc": set_results["val_acc"], "test_acc":set_results["test_acc"]})
    tune.report(set_results)

# Defining Search Space
search_space = {
    # "model_type": tune.choice(["cnn", "mlp", "resnet18", "letnet"]),
    # "batch_size": tune.choice([32, 64, 128, 256, 512]), 
    "lr": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "dropout": tune.uniform(0.2, 0.4)
}

#search_alg = BayesOptSearch(metric="val_acc", mode="max") # bayesian opt maximizing first argument 
search_alg = OptunaSearch(metric="val_acc", mode="max") # optuna opt maximizing first argument 

scheduler = ASHAScheduler(metric="val_acc", mode="max")

# Run Ray Tune
if __name__ == "__main__":
    trainable = tune.with_resources(ray_train, {"cpu": 4, "gpu": 1})
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            # metric="val_acc",
            # mode="max",
            num_samples=100,
            search_alg=search_alg,
            scheduler=scheduler,
        ), 
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="val_acc", mode="max")
    print("Best config:", best_result.config)
    print("Metrics:")
    print(f"Validation Accuracy: {best_result.metrics['val_acc']}")
    print(f"Validation Loss:     {best_result.metrics['val_loss']}")

    # make sure to change model type and batch size each time
    with open("raytune_forest_results.csv", "a") as f:
        f.write(f"{best_result.metrics['val_acc']:.6f},{best_result.metrics['val_loss']:.6f},{type(search_alg).__name__},cnn,512,{best_result.config['lr']},{best_result.config['weight_decay']},{best_result.config['dropout']}\n")