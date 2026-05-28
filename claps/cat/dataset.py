"""
Dataset: sensor time series (pkl tensors) + Generator-LLM cognitive symptoms -> PEPQ-R labels
"""

import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


TEXT_ENCODER = 'nlpai-lab/KURE-v1'


class CognitiveDataset(Dataset):
    def __init__(self, args, flag='TRAIN'):
        self.num_feature = 30
        self.num_items = 9
        self.flag = flag
        self.fold_idx = args.fold_idx
        self.n_folds = args.n_folds

        os.environ['TOKENIZERS_PARALLELISM'] = 'True'
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
        self.use_flat_text = True
        self.flat_text_max_len = getattr(args, 'flat_text_max_len', 4096)

        self.base_dir = args.base_dir
        self.timeseries_dir = args.timeseries_dir
        self.cognitive_json = (getattr(args, 'cognitive_json', None)
                             or os.path.join(self.base_dir, 'cognitive_symptoms.json'))

        self.__read_data__()

        all_scene_lens = [l for counts in self.scene_frame_counts for l in counts]
        self.max_scene_len = max(all_scene_lens)
        self.max_session_len = max(end - start for start, end in self.indices)
        self.max_seq_len = self.max_session_len
        self.seq_len = self.max_session_len
        print(f"  max_scene_len={self.max_scene_len}, max_session_len={self.max_session_len}, max_scenes={self.max_scenes}")

    def __read_data__(self):
        with open(os.path.join(self.timeseries_dir, 'indices.pkl'), 'rb') as f:
            self.indices = pkl.load(f)
        with open(os.path.join(self.timeseries_dir, 'time_series.pkl'), 'rb') as f:
            self.time_series = torch.from_numpy(pkl.load(f).astype(np.float32))
        with open(os.path.join(self.timeseries_dir, 'participants.pkl'), 'rb') as f:
            self.participants = pkl.load(f)
        with open(os.path.join(self.timeseries_dir, 'scene_frame_counts.pkl'), 'rb') as f:
            self.scene_frame_counts = pkl.load(f)

        n_total = len(self.participants)
        n_half = n_total // 2  # 0~31=S1, 32~63=S2

        # --- PEPQ-R labels ---
        excel_path = os.path.join(self.base_dir, 'VRST_dataset_all', '유저스터디 결과정리_final.xlsx')
        raw = pd.read_excel(excel_path, sheet_name='가공', header=None)
        exclude = {17, 24}
        pepq_map = {}
        for i in range(3, 37):
            pid = int(raw.iloc[i, 0])
            if pid in exclude:
                continue
            pepq_map[(pid, 'S1')] = [int(raw.iloc[i, c]) - 1 for c in range(41, 50)]
            pepq_map[(pid, 'S2')] = [int(raw.iloc[i, c]) - 1 for c in range(84, 93)]

        self.labels = np.zeros((n_total, self.num_items), dtype=np.int64)
        for i in range(n_total):
            pid = int(self.participants[i])
            scenario = 'S1' if i < n_half else 'S2'
            self.labels[i] = pepq_map[(pid, scenario)]
        self.labels = torch.from_numpy(self.labels).float()

        # --- cognitive symptom text (Generator-LLM output) ---
        cognitive_path = self.cognitive_json
        print(f"  [cognitive] {cognitive_path}")
        with open(cognitive_path, 'r', encoding='utf-8') as f:
            cognitive_data = json.load(f)

        cog_map = {}
        for entry in cognitive_data:
            if entry.get('status') != 'success':
                continue
            parsed = entry.get('parsed', {})
            text = parsed.get('cognitive symptoms of anxiety', '')
            cog_map[(entry['pid'], entry['scenario'], entry['scene_number'])] = text

        self.scene_texts_raw = []
        for i in range(n_total):
            pid = int(self.participants[i])
            scenario = 'S1' if i < n_half else 'S2'
            n_scenes = len(self.scene_frame_counts[i])
            texts = []
            for s in range(1, n_scenes + 1):
                text = cog_map.get((pid, scenario, s), '')
                if not text:
                    text = f"No cognitive data available for P{pid} {scenario} scene {s}"
                texts.append(text)
            self.scene_texts_raw.append(texts)

        self.max_scenes = max(len(st) for st in self.scene_texts_raw)
        print(f"  max_scenes={self.max_scenes}")

        # tokenize
        all_scene_texts = []
        self.scene_offsets = []
        for sample_scenes in self.scene_texts_raw:
            self.scene_offsets.append((len(all_scene_texts), len(sample_scenes)))
            all_scene_texts.extend(sample_scenes)

        all_tokenized = self.tokenizer(all_scene_texts, padding=True, truncation=True, max_length=512)
        self.all_scene_input_ids = torch.tensor(all_tokenized['input_ids'])
        self.all_scene_attention_mask = torch.tensor(all_tokenized['attention_mask'])
        self.tok_seq_len = self.all_scene_input_ids.shape[1]
        print(f"  tokenized {len(all_scene_texts)} scene texts (tok_len={self.tok_seq_len})")

        # For long-context models, tokenize the whole scenario at once (flat path)
        if self.use_flat_text:
            concat_texts = [' '.join(scenes) for scenes in self.scene_texts_raw]
            flat_tok = self.tokenizer(
                concat_texts, padding=True, truncation=True,
                max_length=self.flat_text_max_len,
            )
            self.flat_input_ids = torch.tensor(flat_tok['input_ids'])
            self.flat_attention_mask = torch.tensor(flat_tok['attention_mask'])
            self.flat_tok_len = self.flat_input_ids.shape[1]
            print(f"  tokenized concatenated scenario text (flat_tok_len={self.flat_tok_len}, max={self.flat_text_max_len})")

        # --- participant-level K-fold split ---
        unique_participants = sorted(set(self.participants.tolist()))
        n_participants = len(unique_participants)

        rng = np.random.RandomState(2026)
        shuffled_p = list(rng.permutation(unique_participants))

        fold_size = n_participants // self.n_folds
        test_start = self.fold_idx * fold_size
        test_end = test_start + fold_size
        test_p = set(shuffled_p[test_start:test_end])
        train_p = set(shuffled_p) - test_p

        train_idx, test_idx = [], []
        for i in range(len(self.indices)):
            pid = self.participants[i]
            if pid in train_p:
                train_idx.append(i)
            else:
                test_idx.append(i)

        if self.flag == 'TRAIN':
            self.idx = np.array(train_idx)
        elif self.flag in ('VAL', 'TEST'):
            self.idx = np.array(test_idx)
        elif self.flag == 'ALL':
            self.idx = np.arange(len(self.indices))

        print(f"  [{self.flag}] samples={len(self.idx)}, train_p={len(train_p)}, test_p={len(test_p)}")

        if len(train_idx) > 0:
            train_labels = self.labels[train_idx].flatten().numpy().astype(np.int64)
            self.class_freq = np.bincount(train_labels, minlength=7)
        else:
            self.class_freq = np.ones(7)

    def __getitem__(self, index):
        i = self.idx[index]
        start_frame, end_frame = self.indices[i]
        scene_counts = self.scene_frame_counts[i]
        num_scenes = len(scene_counts)

        # text
        offset, n_sc = self.scene_offsets[i]
        input_ids = self.all_scene_input_ids[offset:offset + n_sc]
        attention_mask = self.all_scene_attention_mask[offset:offset + n_sc]

        if n_sc < self.max_scenes:
            pad_len = self.max_scenes - n_sc
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, self.tok_seq_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, self.tok_seq_len, dtype=torch.long)])

        seq_x_text = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'num_scenes': torch.tensor(num_scenes, dtype=torch.long),
        }
        if self.use_flat_text:
            seq_x_text['flat_input_ids'] = self.flat_input_ids[i]
            seq_x_text['flat_attention_mask'] = self.flat_attention_mask[i]

        raw_data = self.time_series[start_frame:end_frame]

        # Flat path
        real_len = raw_data.shape[0]
        if real_len >= self.max_session_len:
            seq_x_time_flat = raw_data[:self.max_session_len]
            flat_real_len = self.max_session_len
        else:
            pad_len = self.max_session_len - real_len
            padding = raw_data[-1].unsqueeze(0).repeat(pad_len, 1)
            seq_x_time_flat = torch.cat([raw_data, padding], dim=0)
            flat_real_len = real_len

        # Scene path
        seq_x_time_scene = torch.zeros(self.max_scenes, self.max_scene_len, self.num_feature)
        scene_real_lens = torch.zeros(self.max_scenes, dtype=torch.long)

        offset_frame = 0
        for s_idx, s_len in enumerate(scene_counts):
            scene_data = raw_data[offset_frame:offset_frame + s_len]
            scene_real_lens[s_idx] = s_len
            if s_len >= self.max_scene_len:
                seq_x_time_scene[s_idx] = scene_data[:self.max_scene_len]
                scene_real_lens[s_idx] = self.max_scene_len
            else:
                seq_x_time_scene[s_idx, :s_len] = scene_data
                seq_x_time_scene[s_idx, s_len:] = scene_data[-1].unsqueeze(0).expand(self.max_scene_len - s_len, -1)
            offset_frame += s_len

        seq_y = self.labels[i]

        return (seq_x_time_flat, seq_x_time_scene, seq_x_text, seq_y,
                torch.tensor(flat_real_len, dtype=torch.long),
                scene_real_lens, torch.tensor(num_scenes, dtype=torch.long))

    def __len__(self):
        return len(self.idx)
