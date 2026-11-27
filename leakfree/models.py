"""The models, retrained without leakage.

The boosting models keep the hyper-parameters of the archived run so that the
only difference between the old and the new numbers is the split and the
preprocessing.  The cGAN keeps the architecture of
``D:\\CatBoost\\<TICKER>\\new_cat_conGAN_modified.py`` (EnhancedGenerator /
EnhancedDiscriminator) and changes two things: it is trained on the training
windows only, and the generator that is kept is the one with the best
validation loss, never the one with the best test loss.
"""
from __future__ import annotations

import numpy as np

from . import config

# --------------------------------------------------------------------------- #
# boosting models
# --------------------------------------------------------------------------- #
def train_catboost(x_train, y_train, x_val, y_val, seed=config.DEFAULT_SEED):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.01,
        depth=6,
        subsample=0.8,
        colsample_bylevel=0.8,
        min_data_in_leaf=1,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=50,
        use_best_model=True,
        border_count=128,
        l2_leaf_reg=3,
        bagging_temperature=1,
    )
    # early stopping is driven by the validation block, not by the test block
    model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
    return model, {"best_iteration": int(getattr(model, "best_iteration_", 0) or 0)}


def train_lightgbm(x_train, y_train, x_val, y_val, seed=config.DEFAULT_SEED):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=config.DEFAULT_LIGHTGBM_ROUNDS,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=seed,
        verbose=-1,
    )
    model.fit(x_train, y_train,
              eval_set=[(x_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, {"best_iteration": int(getattr(model, "best_iteration_", 0) or 0)}


def train_xgboost(x_train, y_train, x_val, y_val, seed=config.DEFAULT_SEED):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=config.DEFAULT_XGB_ROUNDS,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=seed,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model, {"best_iteration": int(getattr(model, "best_iteration", 0) or 0)}


# --------------------------------------------------------------------------- #
# conditional GAN
# --------------------------------------------------------------------------- #
def _build_torch_modules(window: int):
    import torch
    from torch import nn

    class EnhancedGenerator(nn.Module):
        """Architecture copied from the archived training script."""

        def __init__(self, noise_dim=100, condition_dim=window, hidden_dim=256, output_dim=1):
            super().__init__()
            self.noise_dim = noise_dim
            self.condition_dim = condition_dim
            self.hidden_dim = hidden_dim
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.LeakyReLU(0.2),
            )
            self.generator = nn.Sequential(
                nn.Linear(noise_dim + hidden_dim // 2, hidden_dim * 3),
                nn.BatchNorm1d(hidden_dim * 3),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Sigmoid(),
            )

        def forward(self, noise, condition):
            cond = self.condition_encoder(condition)
            combined = torch.cat([noise, cond], dim=1)
            return self.generator(combined)

    class EnhancedDiscriminator(nn.Module):
        """Architecture copied from the archived training script."""

        def __init__(self, input_dim=1, condition_dim=window, hidden_dim=256):
            super().__init__()
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
            )
            self.discriminator = nn.Sequential(
                nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x, condition):
            cond = self.condition_encoder(condition)
            combined = torch.cat([x, cond], dim=1)
            return self.discriminator(combined)

    return torch, EnhancedGenerator, EnhancedDiscriminator


def train_cgan(x_train, y_train_scaled, x_val, y_val_scaled, window: int,
               seed: int = config.DEFAULT_SEED,
               epochs: int = config.DEFAULT_CGAN_EPOCHS,
               batch_size: int = config.DEFAULT_CGAN_BATCH,
               noise_dim: int = 100, hidden_dim: int = 256):
    """Train the conditional GAN and keep the generator with the best validation loss."""
    import torch
    from torch import nn

    torch, Generator, Discriminator = _build_torch_modules(window)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim=noise_dim, condition_dim=window,
                          hidden_dim=hidden_dim).to(device)
    discriminator = Discriminator(input_dim=1, condition_dim=window,
                                  hidden_dim=hidden_dim).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_scaled.reshape(-1, 1), dtype=torch.float32, device=device)

    n = len(x_train_t)
    if n < 4:
        raise ValueError("not enough training samples for the GAN")

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in generator.state_dict().items()}
    best_epoch = 0
    history = []

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        permutation = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            if len(idx) < 2:
                continue
            real = y_train_t[idx]
            condition = x_train_t[idx]
            noise = torch.randn(len(idx), noise_dim, device=device)

            # discriminator
            fake = generator(noise, condition).detach()
            d_real = discriminator(real, condition)
            d_fake = discriminator(fake, condition)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + \
                criterion(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # generator
            fake = generator(noise, condition)
            d_fake = discriminator(fake, condition)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        # validation loss of the generator: the only selection criterion used
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(len(x_val_t), noise_dim, device=device)
            pred = generator(noise, x_val_t).cpu().numpy().flatten()
        val_rmse = float(np.sqrt(np.mean((pred - y_val_scaled) ** 2)))
        history.append({"epoch": epoch, "val_rmse": val_rmse, "d_loss": float(d_loss.item())})
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in generator.state_dict().items()}

    generator.load_state_dict(best_state)
    generator.eval()
    return generator, {"best_epoch": best_epoch, "best_val_rmse": best_val,
                       "epochs": epochs, "device": str(device), "history": history}


def predict_cgan(generator, x_eval, seed: int = config.DEFAULT_SEED, noise_dim: int = 100):
    """Forecast with the selected generator.

    The noise vector is fixed so that the forecast is deterministic and can be
    reproduced exactly from the saved seed.
    """
    import torch

    device = next(generator.parameters()).device
    torch.manual_seed(seed)
    x = torch.tensor(x_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        noise = torch.randn(len(x), noise_dim, device=device)
        out = generator(noise, x).cpu().numpy().flatten()
    return out


# --------------------------------------------------------------------------- #
# benchmarks
# --------------------------------------------------------------------------- #
def random_walk_prediction(window_data, idx):
    """The forecast of the next close is the last observed close."""
    return window_data.x[idx][:, -1]


def buy_and_hold_return(window_data, idx):
    """Buy-and-hold over the test block: simple returns of the price itself."""
    closes = window_data.y[idx]
    prev = np.concatenate([[window_data.x[idx][0, -1]], closes[:-1]])
    return closes / prev - 1.0
