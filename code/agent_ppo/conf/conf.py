#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
清扫大作战 PPO 配置。
"""


class Config:

    # Feature dimensions
    # 特征维度
    FEATURES = [
        2 * 7 * 7, # local view: dual-channel avg pool (obstacle/dirt) 21×21→7×7×2=98
        9,         # global state (6 base + battery_max + n_chargers + n_npcs)
        8,         # legal action
        7,         # charger feats (nearest: dist+dx+dz, target: dist+dx+dz, urgency)
        4,         # npc feats
        11,        # traj feats (K=1 => 8 + dx/dz + loop)
        5,         # memory feats
        6,         # bfs feats (charger_dist + dirt_dist + dirt_dx + dirt_dz + reach_ratio + urgency)
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value
    # 单头价值
    VALUE_NUM = 1

    # PPO hyperparameters
    # PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.005          # entropy coef start (moderate exploration) / 熵系数起始值（适度探索）
    BETA_END = 0.001            # entropy coef end (low for late convergence) / 熵系数终值（收敛）
    BETA_DECAY_STEPS = 50000    # steps to decay from BETA_START to BETA_END / 衰减步数
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
