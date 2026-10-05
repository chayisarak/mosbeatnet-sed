from __future__ import annotations
import os
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import config
from experiments import EXP_LIST

TARGET_MODEL = 'mosbeatnet_v2'
BASELINES = ['mosqplus', 'sednet', 'cfresnet1d_small', 'mtrcnn']
MODELS = [TARGET_MODEL] + BASELINES
SEEDS = config.THESIS_TRAINING_SEEDS
CHECKPOINT = 'best_macro_f1'
ALPHA = 0.05
N_PERM = 10000
PERM_SEED = 42

SAVE_DIR = os.path.join(config.OUTPUT_BASE, 'statistical_tests')


def find_latest_run(exp_name, model, seed):
    seed_dir = os.path.join(config.OUTPUT_BASE, exp_name, model, f'seed_{seed}')
    if not os.path.isdir(seed_dir): raise FileNotFoundError(seed_dir)

    runs = []
    for name in os.listdir(seed_dir):
        run_dir = os.path.join(seed_dir, name)
        eval_dir = os.path.join(run_dir, 'evaluation', CHECKPOINT)
        if name.startswith('run_') and os.path.isdir(eval_dir): runs.append(run_dir)

    if not runs: raise FileNotFoundError(f'No evaluated run for {exp_name}, {model}, seed={seed}')
    return max(runs, key=os.path.getmtime)


def load_predictions(path):
    cols = ['file_name', 'segment_index', 'true_label', 'predicted_label']
    df = pd.read_csv(path, usecols=cols)

    df['file_name'] = df['file_name'].astype(str).str.replace('\\', '/', regex=False)
    df['segment_index'] = df['segment_index'].astype(int)
    df['true_label'] = df['true_label'].astype(str)
    df['predicted_label'] = df['predicted_label'].astype(str)
    df['sample_id'] = df['file_name'] + '::seg_' + df['segment_index'].astype(str)
    df['correct'] = df['true_label'] == df['predicted_label']

    if df['sample_id'].duplicated().any(): raise ValueError(f'Duplicate sample in {path}')
    return df.set_index('sample_id')


def get_task(df):
    labels = {x for x in df['true_label'].astype(str) if x.lower() != 'noise'}
    if not labels: return 'Noise-only'

    has_sex = any(x.endswith(('_F', '_M', '.F', '.M')) for x in labels)
    return 'Species+Sex' if has_sex else 'Species'


def align_models(data):
    ref = data[TARGET_MODEL]

    for model in BASELINES:
        df = data[model]

        if set(df.index) != set(ref.index):
            raise ValueError(f'Sample mismatch: {TARGET_MODEL} vs {model}')

        df = df.loc[ref.index]

        if not np.array_equal(ref['true_label'].to_numpy(), df['true_label'].to_numpy()):
            raise ValueError(f'Ground truth mismatch: {TARGET_MODEL} vs {model}')

        data[model] = df

    return data


def save_numpy(data, exp_name, seed, eval_set):
    # save_dir = os.path.join(
    #     config.OUTPUT_BASE, exp_name,
    #     'statistical_tests', CHECKPOINT,
    #     f'seed_{seed}', 'numpy'
    # )

    save_dir = os.path.join(
    SAVE_DIR,
    'numpy',
    exp_name,
    CHECKPOINT,
    f'seed_{seed}'
    )  # output/statistical_tests/numpy/<exp_name>/<checkpoint>/seed_<seed>/
    os.makedirs(save_dir, exist_ok=True)

    ref = data[TARGET_MODEL]
    arrays = {
        'sample_id': ref.index.to_numpy(),
        'file_name': ref['file_name'].to_numpy(),
        'segment_index': ref['segment_index'].to_numpy(),
        'true_label': ref['true_label'].to_numpy()
    }

    for model in MODELS:
        arrays[f'{model}_pred'] = data[model]['predicted_label'].to_numpy()
        arrays[f'{model}_correct'] = data[model]['correct'].to_numpy(dtype=bool)

    path = os.path.join(save_dir, f'{eval_set}.npz')
    np.savez_compressed(path, **arrays)
    return path


#=============== Segment-level McNemar test ===============

def segment_mcnemar(data):
    mos = data[TARGET_MODEL]['correct'].to_numpy(dtype=bool)
    rows = []

    for baseline in BASELINES:
        base = data[baseline]['correct'].to_numpy(dtype=bool)

        n11 = int(np.sum(mos & base))
        n10 = int(np.sum(mos & ~base))
        n01 = int(np.sum(~mos & base))
        n00 = int(np.sum(~mos & ~base))
        discordant = n10 + n01
        table = [[n11, n10], [n01, n00]]

        if discordant == 0:
            stat, p, method = 0.0, 1.0, 'No discordant pairs'
        elif discordant < 25:
            test = mcnemar(table, exact=True)
            stat, p, method = float(test.statistic), float(test.pvalue), 'Exact'
        else:
            test = mcnemar(table, exact=False, correction=True)
            stat, p, method = float(test.statistic), float(test.pvalue), 'Chi-square corrected'

        rows.append({
            'Baseline': baseline,
            'n10': n10,
            'n01': n01,
            'Discordant': discordant,
            'Method': method,
            'Statistic': stat,
            'pRaw': p
        })

    result = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(result['pRaw'], alpha=ALPHA, method='holm')

    result['pHolm'] = p_holm
    result['Significant'] = reject

    result['Result'] = [
        'No significant difference' if not sig else
        'MosBeatNet better' if n10 > n01 else
        'Baseline better'
        for sig, n10, n01 in zip(result['Significant'], result['n10'], result['n01'])
    ]

    return result


#=============== File-level paired permutation test ===============

def file_level_test(data):
    mos_file = data[TARGET_MODEL].groupby('file_name')['correct'].mean()
    rows = []
    rng = np.random.default_rng(PERM_SEED)

    for baseline in BASELINES:
        base_file = data[baseline].groupby('file_name')['correct'].mean()
        common = mos_file.index.intersection(base_file.index)

        mos = mos_file.loc[common].to_numpy(dtype=float)
        base = base_file.loc[common].to_numpy(dtype=float)
        diff = mos - base

        observed = float(diff.mean())

        perm_values = np.zeros(N_PERM)
        for i in range(N_PERM):
            signs = rng.choice([-1.0, 1.0], size=len(diff))
            perm_values[i] = np.mean(diff * signs)

        p = (np.sum(np.abs(perm_values) >= abs(observed)) + 1) / (N_PERM + 1)

        rows.append({
            'Baseline': baseline,
            'Files': len(common),
            'MosBeatNet File Accuracy': float(mos.mean()),
            'Baseline File Accuracy': float(base.mean()),
            'Mean Difference': observed,
            'Difference pp': observed * 100,
            'pRaw': float(p)
        })

    result = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(result['pRaw'], alpha=ALPHA, method='holm')

    result['pHolm'] = p_holm
    result['Significant'] = reject

    result['Result'] = [
        'No significant difference' if not sig else
        'MosBeatNet better' if diff > 0 else
        'Baseline better'
        for sig, diff in zip(result['Significant'], result['Mean Difference'])
    ]

    return result


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    all_mcnemar = []
    all_file_tests = []

    for exp_key, exp_cfg in EXP_LIST.items():
        exp_name = exp_cfg.get('exp_name', exp_key)

        for seed in SEEDS:
            print()
            print('=' * 70)
            print(f'{exp_name} | seed={seed}')
            print('=' * 70)

            try:
                run_dirs = {model: find_latest_run(exp_name, model, seed) for model in MODELS}
            except FileNotFoundError as e:
                print(f'Skip: {e}')
                continue

            common_eval_sets = None

            for model in MODELS:
                eval_root = os.path.join(run_dirs[model], 'evaluation', CHECKPOINT)
                eval_sets = {
                    name for name in os.listdir(eval_root)
                    if os.path.isfile(os.path.join(eval_root, name, 'segment_predictions.csv'))
                }

                common_eval_sets = eval_sets if common_eval_sets is None else common_eval_sets & eval_sets

            if not common_eval_sets:
                print('No common evaluation sets')
                continue

            for eval_set in sorted(common_eval_sets):
                data = {}

                for model in MODELS:
                    path = os.path.join(
                        run_dirs[model], 'evaluation', CHECKPOINT,
                        eval_set, 'segment_predictions.csv'
                    )
                    data[model] = load_predictions(path)

                data = align_models(data)
                task = get_task(data[TARGET_MODEL])

                seg_counts = data[TARGET_MODEL].groupby('file_name').size()
                if not (seg_counts == 20).all():
                    print(f'Warning: {eval_set} has files with segment count != 20')

                save_numpy(data, exp_name, seed, eval_set)

                # Segment-level McNemar
                mc = segment_mcnemar(data)
                mc.insert(0, 'Task', task)
                mc.insert(0, 'Test Set', eval_set)
                mc.insert(0, 'Seed', seed)
                mc.insert(0, 'Experiment', exp_name)
                all_mcnemar.append(mc)

                # File-level paired permutation
                ft = file_level_test(data)
                ft.insert(0, 'Task', task)
                ft.insert(0, 'Test Set', eval_set)
                ft.insert(0, 'Seed', seed)
                ft.insert(0, 'Experiment', exp_name)
                all_file_tests.append(ft)

                print(f'{eval_set} | {task}')

    if not all_mcnemar:
        print('No statistical results found')
        return

    mcnemar_df = pd.concat(all_mcnemar, ignore_index=True)
    file_df = pd.concat(all_file_tests, ignore_index=True)

    mcnemar_path = os.path.join(SAVE_DIR, 'segment_mcnemar.csv')
    file_path = os.path.join(SAVE_DIR, 'file_level_permutation.csv')

    mcnemar_df.to_csv(mcnemar_path, index=False)
    file_df.to_csv(file_path, index=False)

    print()
    print('=' * 70)
    print('DONE')
    print('=' * 70)
    print(mcnemar_path)
    print(file_path)


if __name__ == '__main__':
    main()