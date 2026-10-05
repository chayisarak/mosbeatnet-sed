from __future__ import annotations

from collections.abc import Sequence

import config
import data_paths


SOURCE_BACKGROUNDS = {
    config.INDOOR_SOURCE: config.ENVS,
    config.OUTDOOR_SOURCE: ("gaussian",),
    config.HUMBUG_SOURCE: config.ENVS,
    config.BIODCASE_SOURCE: config.ENVS,
}


def source_csvs(source: str, split: str, target_type: str | None = None,
                backgrounds: Sequence[str] | None = None) -> list[str]:
    if backgrounds is None:
        backgrounds = SOURCE_BACKGROUNDS[source]
    return data_paths.metadata_paths(source, tuple(backgrounds), split, target_type=target_type)


def make_eval_sets(model_target: str) -> dict[str, dict]:
    return {
        "indoor_urban": {
            "csvs": source_csvs(config.INDOOR_SOURCE, "test", backgrounds=("urban",)),
            "label_space": model_target,
        },
        "indoor_forest": {
            "csvs": source_csvs(config.INDOOR_SOURCE, "test", backgrounds=("forest",)),
            "label_space": model_target,
        },
        "outdoor_gaussian": {
            "csvs": source_csvs(config.OUTDOOR_SOURCE, "test", backgrounds=("gaussian",)),
            "label_space": model_target,
        },
        "humbug_urban": {
            "csvs": source_csvs(config.HUMBUG_SOURCE, "test", backgrounds=("urban",)),
            "label_space": "species",
        },
        "humbug_forest": {
            "csvs": source_csvs(config.HUMBUG_SOURCE, "test", backgrounds=("forest",)),
            "label_space": "species",
        },
        "biodcase_urban": {
            "csvs": source_csvs(config.BIODCASE_SOURCE, "test", backgrounds=("urban",)),
            "label_space": "species",
        },
        "biodcase_forest": {
            "csvs": source_csvs(config.BIODCASE_SOURCE, "test", backgrounds=("forest",)),
            "label_space": "species",
        },
        "noise_only": {
            "csvs": data_paths.noise_only_metadata_paths(),
            "label_space": model_target,
        },
    }


def make_experiment(*, exp_name: str, description: str, target_type: str, label_order: Sequence[str],
                    train_sources: Sequence[str], sweep_tag: str, allow_missing_train_labels: bool = False,
                    missing_train_labels: Sequence[str] = (), use_weighted_sampler: bool = False) -> dict:
    train_csvs = [path for source in train_sources for path in source_csvs(source, "train", target_type)]
    val_csvs = [path for source in train_sources for path in source_csvs(source, "val", target_type)]

    return {
        **config.TRAIN_CFG,
        **config.EVAL_CFG,
        "exp_name": exp_name,
        "description": description,
        "target_type": target_type,
        "label_order": list(label_order),
        "train_sources": list(train_sources),
        "train_csvs": train_csvs,
        "val_csvs": val_csvs,
        "eval_sets": make_eval_sets(target_type),
        "allow_missing_train_labels": bool(allow_missing_train_labels),
        "missing_train_labels": list(missing_train_labels),
        "sweep_tag": sweep_tag,
        "aug_mode": config.AUG_MODE,
        "wingbeat_name": "miru",
        "report_species_order": list(config.SPECIES_ORDER),
        "use_weighted_sampler": use_weighted_sampler,
    }


EXP1 = make_experiment(
    exp_name="xdomain_exp1_indoor_sx9",
    description="Train/val on Indoor MIRU with species-sex labels.",
    target_type="species_sex",
    label_order=config.SPECIES_SEX_LABEL_ORDER,
    train_sources=(config.INDOOR_SOURCE,),
    sweep_tag="indoor_urban",
)

EXP2 = make_experiment(
    exp_name="xdomain_exp2_outdoor_obs8_sx9",
    description="Train/val on Outdoor MIRU with 9-class output; Cx.Quin_F is missing.",
    target_type="species_sex",
    label_order=config.SPECIES_SEX_LABEL_ORDER,
    train_sources=(config.OUTDOOR_SOURCE,),
    sweep_tag="outdoor_gaussian",
    allow_missing_train_labels=True,
    missing_train_labels=config.OUTDOOR_MISSING_LABELS,
)

EXP3 = make_experiment(
    exp_name="xdomain_exp3_humbug_sp5",
    description="Train/val on HumBug MIRU with species labels.",
    target_type="species",
    label_order=config.SPECIES_LABEL_ORDER,
    train_sources=(config.HUMBUG_SOURCE,),
    sweep_tag="humbug_urban",
)

EXP4 = make_experiment(
    exp_name="xdomain_exp4_biodcase_sp5",
    description="Train/val on BioDCASE with species labels.",
    target_type="species",
    label_order=config.SPECIES_LABEL_ORDER,
    train_sources=(config.BIODCASE_SOURCE,),
    sweep_tag="biodcase_urban",
)

EXP5 = make_experiment(
    exp_name="xdomain_exp5_indoor_outdoor_sx9",
    description="Train/val on Indoor + Outdoor MIRU with species-sex labels.",
    target_type="species_sex",
    label_order=config.SPECIES_SEX_LABEL_ORDER,
    train_sources=(config.INDOOR_SOURCE, config.OUTDOOR_SOURCE),
    sweep_tag="indoor_urban",
)

EXP6 = make_experiment(
    exp_name="xdomain_exp6_all_sources_sp5",
    description="Train/val on all sources with species labels.",
    target_type="species",
    label_order=config.SPECIES_LABEL_ORDER,
    train_sources=config.MOSQUITO_SOURCES,
    sweep_tag="indoor_urban",
)

EXP1_WS = make_experiment(
    exp_name="xdomain_exp1_indoor_sx9_ws",
    description="Train/val on Indoor MIRU with species-sex labels.",
    target_type="species_sex",
    label_order=config.SPECIES_SEX_LABEL_ORDER,
    train_sources=(config.INDOOR_SOURCE,),
    sweep_tag="indoor_urban",
    use_weighted_sampler=True
)

EXP2_WS = make_experiment(
    exp_name="xdomain_exp2_outdoor_obs8_sx9_ws",
    description="Train/val on Outdoor MIRU with 9-class output; Cx.Quin_F is missing.",
    target_type="species_sex",
    label_order=config.SPECIES_SEX_LABEL_ORDER,
    train_sources=(config.OUTDOOR_SOURCE,),
    sweep_tag="outdoor_gaussian",
    allow_missing_train_labels=True,
    missing_train_labels=config.OUTDOOR_MISSING_LABELS,
    use_weighted_sampler=True
)

EXP3_WS = make_experiment(
    exp_name="xdomain_exp3_humbug_sp5_ws",
    description="Train/val on HumBug MIRU with species labels.",
    target_type="species",
    label_order=config.SPECIES_LABEL_ORDER,
    train_sources=(config.HUMBUG_SOURCE,),
    sweep_tag="humbug_urban",
    use_weighted_sampler=True
)

EXP4_WS = make_experiment(
    exp_name="xdomain_exp4_biodcase_sp5_ws",
    description="Train/val on BioDCASE with species labels.",
    target_type="species",
    label_order=config.SPECIES_LABEL_ORDER,
    train_sources=(config.BIODCASE_SOURCE,),
    sweep_tag="biodcase_urban",
    use_weighted_sampler=True
)

EXP5_WS = make_experiment(
    exp_name="xdomain_exp5_indoor_outdoor_sx9_ws",
    description="Train/val on Indoor + Outdoor MIRU with species-sex labels.",
    target_type="species_sex",
    label_order=config.SPECIES_SEX_LABEL_ORDER,
    train_sources=(config.INDOOR_SOURCE, config.OUTDOOR_SOURCE),
    sweep_tag="indoor_urban",
    use_weighted_sampler=True
)

EXP6_WS = make_experiment(
    exp_name="xdomain_exp6_all_sources_sp5_ws",
    description="Train/val on all sources with species labels.",
    target_type="species",
    label_order=config.SPECIES_LABEL_ORDER,
    train_sources=config.MOSQUITO_SOURCES,
    sweep_tag="indoor_urban",
    use_weighted_sampler=True
)

EXP_LIST = {
    "e1_indoor_sx9": EXP1,
    "e2_outdoor_sx9": EXP2,
    "e3_humbug_sp5": EXP3,
    "e4_biodcase_sp5": EXP4,
    "e5_indoor_outdoor_sx9": EXP5,
    "e6_all_sources_sp5": EXP6,


    "e1_indoor_sx9_ws": EXP1_WS,
    "e2_outdoor_sx9_ws": EXP2_WS,
    "e3_humbug_sp5_ws": EXP3_WS,
    "e4_biodcase_sp5_ws": EXP4_WS,
    "e5_indoor_outdoor_sx9_ws": EXP5_WS,
    "e6_all_sources_sp5_ws": EXP6_WS,
}
