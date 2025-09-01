#thanks to perplexity.ai
from ax.service.managed_loop import optimize

def train_evaluate(params):
    lr = params["learning_rate"]
    momentum = params["momentum"]
    # Here, insert PyTorch model training and validation logic using lr and momentum.
    # For demo, return a fake validation accuracy:
    val_accuracy = 0.8 + 0.1 * (1 - abs(lr - 0.01)) - 0.05 * abs(momentum - 0.9)
    return val_accuracy

best_params, best_vals, experiment, model = optimize(
    parameters=[
        {"name": "learning_rate", "type": "range", "bounds": [0.001, 0.1]},
        {"name": "momentum", "type": "range", "bounds": [0.7, 0.99]},
    ],
    evaluation_function=train_evaluate,
    objective_name='val_accuracy',
    total_trials=20,
)

print("Best parameters:", best_params)
print("Best validation accuracy:", best_vals)
