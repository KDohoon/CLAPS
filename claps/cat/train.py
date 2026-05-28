"""
Training / evaluation for the Cross-modal Aligned Transformer (CAT).

Usage:
  python -m claps.cat.train                       # all 8 folds
  python -m claps.cat.train --fold_idx 0          # single fold
  python -m claps.cat.train --ablation path1_only # ablation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import random
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from claps.cat.model import DualPathEncoder
from claps.cat.dataset import CognitiveDataset


# ============================================================
# Helpers
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def collate_fn(batch):
    flat_list, scene_list, text_list, label_list, flat_rl_list, scene_rl_list, ns_list = zip(*batch)

    text = {
        'input_ids': torch.stack([t['input_ids'] for t in text_list]),
        'attention_mask': torch.stack([t['attention_mask'] for t in text_list]),
        'num_scenes': torch.stack([t['num_scenes'] for t in text_list]),
    }
    if 'flat_input_ids' in text_list[0]:
        text['flat_input_ids'] = torch.stack([t['flat_input_ids'] for t in text_list])
        text['flat_attention_mask'] = torch.stack([t['flat_attention_mask'] for t in text_list])

    return (torch.stack(flat_list), torch.stack(scene_list), text,
            torch.stack(label_list), torch.stack(flat_rl_list),
            torch.stack(scene_rl_list), torch.stack(ns_list))


def evaluate(model, data_loader, criterion, device):
    trues, preds, total_loss = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            batch_x_flat, batch_x_scene, batch_x_text, label, flat_rl, scene_rl, ns = batch
            batch_x_flat = batch_x_flat.float().to(device)
            batch_x_scene = batch_x_scene.float().to(device)
            label = label.float().to(device)

            outputs, _ = model(batch_x_flat, batch_x_scene, batch_x_text, flat_rl, scene_rl, ns)
            loss = criterion(outputs, label)
            total_loss.append(loss.item())
            trues.append(label.detach().cpu())
            preds.append(outputs.detach().cpu())

    trues = torch.cat(trues, 0)
    preds = torch.cat(preds, 0)

    mae = torch.mean(torch.abs(preds - trues)).item()
    mse = torch.mean((preds - trues) ** 2).item()

    preds_cls = preds.clamp(0, 6).round().long().flatten().numpy()
    trues_cls = trues.long().flatten().numpy()
    f1_micro = f1_score(trues_cls, preds_cls, average='micro')
    f1_macro = f1_score(trues_cls, preds_cls, average='macro')

    model.train()
    return np.average(total_loss), mae, mse, f1_micro, f1_macro


# ============================================================
# Train one fold
# ============================================================

def train_one_fold(args, fold_idx):
    args.fold_idx = fold_idx
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f'\n{"="*60}')
    print(f'  Fold {fold_idx} | Device: {device}')
    print(f'{"="*60}')

    train_data = CognitiveDataset(args, flag='TRAIN')
    test_data = CognitiveDataset(args, flag='TEST')

    args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
    args.enc_in = train_data.num_feature

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn, drop_last=False)

    model = DualPathEncoder(args, device).float().to(device)

    # Exclude frozen params (e.g. the text encoder) to save AdamW state memory
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.SmoothL1Loss()

    # Run/setting name
    ablation_str = f'_{args.ablation}' if args.ablation != 'none' else ''
    setting = 'cognitive{}_kure_seed{}_fold{}_el{}_dim{}_ds{}_lr{}_bs{}_do{}_dff{}_nh{}_pl{}_st{}'.format(
        ablation_str, args.seed, fold_idx,
        args.e_layers, args.d_model, args.d_scene,
        args.learning_rate, args.batch_size, args.dropout,
        args.d_ff, args.n_heads, args.patch_len, args.stride
    )

    ckpt_path = os.path.join(args.checkpoints, setting)
    results_path = os.path.join('results', setting)

    if os.path.exists(os.path.join(results_path, 'pred_raw.npy')):
        print(f'Results already exist for {setting}, skipping.')
        return None

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.train_epochs}'):
            batch_x_flat, batch_x_scene, batch_x_text, label, flat_rl, scene_rl, ns = batch
            if len(label) <= 1:
                continue

            batch_x_flat = batch_x_flat.float().to(device)
            batch_x_scene = batch_x_scene.float().to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_x_flat, batch_x_scene, batch_x_text, flat_rl, scene_rl, ns)
            loss = criterion(outputs, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            train_loss.append(loss.item())

        print(f"\nEpoch {epoch+1} cost time: {time.time() - epoch_time:.1f}s")

        train_loss_avg = np.average(train_loss)
        # Model selection (checkpoint + LR schedule) uses the TRAIN loss only.
        # The TEST set is held out and never influences which epoch is chosen.
        test_loss, test_mae, test_mse, test_micro, test_macro = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.3f} Test Loss: {test_loss:.3f} "
              f"MAE: {test_mae:.3f} F1-Micro: {test_micro:.3f} F1-Macro: {test_macro:.3f}")

        early_stopping(train_loss_avg, model, ckpt_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(train_loss_avg)

    # --- Final evaluation with the best model (TEST set) ---
    model.load_state_dict(torch.load(os.path.join(ckpt_path, 'checkpoint.pth')))

    eval_data = CognitiveDataset(args, flag='TEST')
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn, drop_last=False)

    embs, trues, preds_raw = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Final evaluation (TEST)'):
            batch_x_flat, batch_x_scene, batch_x_text, label, flat_rl, scene_rl, ns = batch
            batch_x_flat = batch_x_flat.float().to(device)
            batch_x_scene = batch_x_scene.float().to(device)
            label = label.float().to(device)

            outputs, emb = model(batch_x_flat, batch_x_scene, batch_x_text, flat_rl, scene_rl, ns)
            trues.append(label.detach().cpu())
            preds_raw.append(outputs.detach().cpu())
            embs.append(emb.detach().cpu())

    trues = torch.cat(trues, 0)
    preds_raw = torch.cat(preds_raw, 0)
    preds = preds_raw.clamp(0, 6).round().long()

    mae = torch.mean(torch.abs(preds_raw - trues)).item()
    mse = torch.mean((preds_raw - trues) ** 2).item()
    trues_cls = trues.long().flatten().numpy()
    preds_cls = preds.flatten().numpy()
    f1_micro = f1_score(trues_cls, preds_cls, average='micro')
    f1_macro = f1_score(trues_cls, preds_cls, average='macro')

    print(f'\n=== Final Results ({setting})  [TEST only] ===')
    print(f'MAE: {mae:.3f}  MSE: {mse:.3f}  RMSE: {np.sqrt(mse):.3f}')
    print(f'F1-Micro: {f1_micro:.3f}  F1-Macro: {f1_macro:.3f}')

    os.makedirs(results_path, exist_ok=True)
    np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, f1_micro, f1_macro]))
    np.save(os.path.join(results_path, 'pred_raw.npy'), preds_raw.numpy())
    np.save(os.path.join(results_path, 'pred.npy'), preds.numpy())
    np.save(os.path.join(results_path, 'true.npy'), trues.numpy())

    embs = torch.cat(embs, 0)
    os.makedirs('embeddings', exist_ok=True)
    with open(os.path.join('embeddings', f'{setting}.pkl'), 'wb') as f:
        pickle.dump(embs, f)

    # --- Free GPU memory (avoid accumulation across folds) ---
    result = {
        'fold': fold_idx,
        'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse),
        'f1_micro': f1_micro, 'f1_macro': f1_macro,
    }
    del model, optimizer, scheduler, early_stopping
    del train_data, test_data, eval_data
    del train_loader, test_loader, eval_loader
    del trues, preds_raw, preds, embs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Path Encoder with Cognitive Text')

    # Paths (see README for the expected data layout)
    parser.add_argument('--base_dir', type=str,
                        default=os.environ.get('CLAPS_DATA_DIR', './data'),
                        help='Root that contains VRST_dataset_all/ (self-report xlsx)')
    parser.add_argument('--timeseries_dir', type=str,
                        default='./data/timeseries',
                        help='Dir with indices.pkl / time_series.pkl / participants.pkl / scene_frame_counts.pkl')
    parser.add_argument('--cognitive_json', type=str,
                        default='./outputs/cognitive_symptoms.json',
                        help='Generator-LLM output JSON (cognitive symptom CS_t per scene)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Model hyperparameters
    parser.add_argument('--flat_text_max_len', type=int, default=4096,
                        help='max_length of the concatenated scenario text (flat path)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--d_scene', type=int, default=128)
    parser.add_argument('--cross_scene_layers', type=int, default=1)
    parser.add_argument('--cross_scene_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--patch_len', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=2026)

    # K-fold
    parser.add_argument('--fold_idx', type=int, default=-1,
                        help='-1 = run all 8 folds')
    parser.add_argument('--n_folds', type=int, default=8)

    # Ablation
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'path1_only', 'path2_only', 'no_cross_scene', 'path2_no_cross_scene'])

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'CUDA available: {torch.cuda.is_available()}')
    args.use_gpu = args.use_gpu and torch.cuda.is_available()

    if args.fold_idx >= 0:
        train_one_fold(args, args.fold_idx)
    else:
        all_results = []
        for fold in range(args.n_folds):
            result = train_one_fold(args, fold)
            if result:
                all_results.append(result)

        if all_results:
            print(f'\n{"="*60}')
            print(f'  K-Fold Results ({len(all_results)} folds)')
            print(f'{"="*60}')
            for r in all_results:
                print(f"  Fold {r['fold']}: MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} "
                      f"F1-Micro={r['f1_micro']:.3f} F1-Macro={r['f1_macro']:.3f}")

            avg_mae = np.mean([r['mae'] for r in all_results])
            avg_rmse = np.mean([r['rmse'] for r in all_results])
            avg_micro = np.mean([r['f1_micro'] for r in all_results])
            avg_macro = np.mean([r['f1_macro'] for r in all_results])
            print(f"\n  Average: MAE={avg_mae:.3f} RMSE={avg_rmse:.3f} "
                  f"F1-Micro={avg_micro:.3f} F1-Macro={avg_macro:.3f}")
