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
        6,         # global state
        8,         # legal action
        4,         # charger feats
        4,         # npc feats
        11,        # traj feats (K=1 => 8 + dx/dz + loop)
        5,         # memory feats (原3 → 5)
        4,         # bfs feats (charger_dist + dirt_dist + reach_ratio + urgency)
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
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
