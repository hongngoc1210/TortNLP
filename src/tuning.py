import optuna
import yaml

from trainer.train_pipeline import train_one_config


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def objective(trial):

    cfg = load_config()

    # search space

    cfg["training"]["lr"] = trial.suggest_float(
        "lr",
        1e-6,
        5e-5,
        log=True
    )

    cfg["training"]["batch_size"] = trial.suggest_categorical(
        "batch_size",
        [2,4,8]
    )

    cfg["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1
    )

    re_f1, td_acc = train_one_config(cfg)

    score = (re_f1 + td_acc) / 2

    return score


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=20)
    
print("Best params:", study.best_params)
print("Best score:", study.best_value)
