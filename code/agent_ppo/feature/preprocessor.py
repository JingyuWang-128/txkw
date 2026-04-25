#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np
from collections import deque


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径
    COARSE_GRID = 16  # coarse visited grid size / 粗粒度访问网格

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 200
        self.battery_max = 200

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # -------- Phase1: entity parsing & history state --------
        self.chargers = []  # list[(x,z)] — current frame chargers
        self.all_known_chargers = set()  # persistent memory of ALL charger positions ever seen
        self.npcs = []  # list[(x,z)]
        self.step_cleaned_cells = []  # list[(x,z)]

        # Stable environment config counts (from env_info, not fluctuating view)
        # 稳定的环境配置计数（来自 env_info，不受视野波动影响）
        self.config_charger_count = 4  # updated from env_info["total_charger"]
        self.config_npc_count = 4      # updated from first-frame NPC list length

        self.last_pos = (0, 0)
        self.last_battery = self.battery

        self.visited = set()  # visited fine cells
        self.visited_coarse = set()  # visited coarse bins
        self.recent_positions = []  # sliding window of recent positions

        self.action_hist_k = 1
        self.action_hist = [-1] * self.action_hist_k

        self.steps_since_clean = 0  # steps since last successful clean / 距上次清扫步数
        self.clean_streak = 0
        self.idle_steps = 0
        self.idle_window = []  # sliding window: 1=idle step, 0=cleaned step
        self.charge_count = 0

        self.last_d_charger_norm = 1.0
        self.last_reward_components = {}
        self.last_coverage_ratio = 0.0  # Track coverage change for reward

        # cached step metrics
        self.d_charger = 200.0
        self.d_npc_min = 200.0
        self.coverage_ratio = 0.0
        self.recent_coverage = 0.0
        self.cleaned_this_step = 0
        self.is_new_cell = False
        self.loop_flag = 0.0
        self.bfs_d_charger = 256.0
        self.bfs_charger_dir = (0, 0)  # first-step direction toward nearest charger / 到最近充电桩的首步方向
        self.bfs_d_dirt = 256.0
        self.bfs_dirt_dir = (0, 0)  # first-step direction toward nearest dirt / 到最近污渍的首步方向
        self.last_d_dirt_norm = 1.0  # previous step's dirt distance (for potential shaping) / 上步污渍距离

        # Raw previous BFS distances for navigational reward delta / 原始BFS距离用于导航奖励增量
        self._prev_bfs_d_dirt = 256.0
        self._prev_bfs_d_charger = 256.0
        self._prev_bfs_d_target = 256.0  # for patrol mode when no dirt visible / 无脏格时巡回模式用
        self.just_charged = False

        # Multi-charger patrol (LRU rotation) / 多充电桩巡回（最少访问优先轮换）
        self.target_charger_pos = None   # the "next" charger to head toward / 下一个目标充电桩位置
        self.last_charged_pos = None     # charger position where agent last charged / 上次充电的充电桩位置
        self.charger_visit_step = {}     # {(x,z): last_visit_step} per charger / 每个充电桩的最后访问步数
        self.bfs_d_target_charger = 256.0
        self.bfs_target_charger_dir = (0, 0)
        self.last_d_target_charger_norm = 1.0

        # Wall collision detection / 撞墙检测
        self.wall_hit = 0.0  # 1.0 if last move was blocked by obstacle / 上次移动被障碍物阻挡

        # Trapped detection / 被困检测
        self.stuck_counter = 0  # consecutive steps with little position change / 连续小范围移动步数
        self.escape_mode = False  # escape mode flag / 突围模式标志
        self.last_escape_pos = (0, 0)  # last position that broke out / 上次突围位置

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.last_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Stable config counts from env_info (frame-independent)
        # 从 env_info 获取稳定的配置计数（帧无关）
        if "total_charger" in env_info:
            self.config_charger_count = int(env_info["total_charger"])

        # Update idle sliding window for _memory_feats / 更新idle滑动窗口
        _step_cleaned = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        self.idle_window.append(0 if _step_cleaned > 0 else 1)
        if len(self.idle_window) > 50:
            self.idle_window = self.idle_window[-50:]

        # Steps since last successful clean / 距上次清扫步数
        if _step_cleaned > 0:
            self.steps_since_clean = 0
        else:
            self.steps_since_clean += 1

        # step_cleaned_cells / 本步清扫格子
        self.step_cleaned_cells = []
        raw_cleaned = env_info.get("step_cleaned_cells") if isinstance(env_info, dict) else None
        if raw_cleaned:
            for c in raw_cleaned:
                if isinstance(c, dict) and "x" in c and "z" in c:
                    self.step_cleaned_cells.append((int(c["x"]), int(c["z"])))
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    self.step_cleaned_cells.append((int(c[0]), int(c[1])))

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        # Entities: chargers(organs sub_type=1) & NPCs
        self.chargers = []
        organs = frame_state.get("organs") if isinstance(frame_state, dict) else None
        if organs:
            for o in organs:
                try:
                    if int(o.get("sub_type", -1)) != 1:
                        continue
                    p = o.get("pos", {})
                    # Position is the top-left of the 3×3 charger area; +1 to get center
                    # OrganState的Position为3×3充电桩左上角坐标，+1修正为中心点
                    self.chargers.append((int(p.get("x", 0)) + 1, int(p.get("z", 0)) + 1))
                except Exception:
                    continue
        # Persist all charger positions ever seen (防止帧数据不全时丢失充电桩)
        for c in self.chargers:
            self.all_known_chargers.add(c)
        # Use comprehensive list for all downstream logic
        # 用完整列表供后续BFS、特征、巡回使用
        self.chargers = list(self.all_known_chargers)

        self.npcs = []
        npcs = frame_state.get("npcs") if isinstance(frame_state, dict) else None
        if npcs:
            for n in npcs:
                try:
                    p = n.get("pos", {})
                    self.npcs.append((int(p.get("x", 0)), int(p.get("z", 0))))
                except Exception:
                    continue
        # Capture NPC count from first frame (all NPCs present at start)
        # 首帧记录NPC数量（开局所有NPC都存在）
        if self.step_no <= 1 and len(self.npcs) > 0:
            self.config_npc_count = len(self.npcs)

        # Wall collision: agent tried to move but stayed in place / 撞墙检测
        self.wall_hit = 1.0 if self.step_no > 0 and self.cur_pos == self.last_pos else 0.0

        # History: action & visited / 历史：动作与访问记忆
        if last_action is None:
            last_action = -1
        self.action_hist = (self.action_hist[1:] + [int(last_action)])[-self.action_hist_k :]

        hx, hz = self.cur_pos
        self.is_new_cell = (hx, hz) not in self.visited
        self.visited.add((hx, hz))

        bin_size = max(1, self.GRID_SIZE // self.COARSE_GRID)
        cx, cz = hx // bin_size, hz // bin_size
        self.visited_coarse.add((int(cx), int(cz)))
        self.coverage_ratio = float(len(self.visited_coarse) / (self.COARSE_GRID * self.COARSE_GRID))

        # recent coverage
        self.recent_positions.append((hx, hz))
        max_recent = 50
        if len(self.recent_positions) > max_recent:
            self.recent_positions = self.recent_positions[-max_recent:]
        uniq_recent = len(set(self.recent_positions))
        self.recent_coverage = float(uniq_recent / max(1, len(self.recent_positions)))

        # loop flag: multi-dimensional loop detection / 回环标志：多维度回环检测
        # 1. Exact repeat: visited same cell in recent 20 steps / 精确回环：20步内重复访问同一格子
        recent_window_short = 20
        recent_short = self.recent_positions[-recent_window_short:] if len(self.recent_positions) >= recent_window_short else self.recent_positions
        loop_exact = 1.0 if len(recent_short) >= 2 and recent_short.count((hx, hz)) >= 2 else 0.0
        
        # 2. Zigzag pattern: high repetition rate in 50-step window / 之字形走位：50步内高重复率
        recent_window_long = 50
        recent_long = self.recent_positions[-recent_window_long:] if len(self.recent_positions) >= recent_window_long else self.recent_positions
        if len(recent_long) > 10:
            unique_count = len(set(recent_long))
            repetition_ratio = 1.0 - (unique_count / len(recent_long))
            loop_zigzag = 1.0 if repetition_ratio > 0.7 else 0.0
        else:
            loop_zigzag = 0.0
        
        # Take maximum of both detections / 取两个检测的最大值
        self.loop_flag = max(loop_exact, loop_zigzag)

        # charge count heuristic: battery increased (e.g., charging)
        self.just_charged = self.battery > self.last_battery + 5
        just_charged = self.just_charged
        if just_charged:
            self.charge_count += 1

        # cache step-level distances
        self.d_charger = self._calc_nearest_entity_dist(self.chargers)
        self.d_npc_min = self._calc_nearest_entity_dist(self.npcs)

        # BFS distances — charger every step (survival-critical), dirt every step
        # BFS距离——充电桩每步算（生死攸关），污渍每步算
        self.bfs_d_charger, self.bfs_charger_dir = self._calc_bfs_to_charger()
        self.bfs_d_dirt, self.bfs_dirt_dir = self._calc_bfs_to_local_dirt()

        # Multi-charger patrol (LRU rotation): visit ALL chargers in turn
        # 多充电桩巡回（LRU轮换）：按最久未访问优先，确保所有充电桩都被巡回
        if just_charged and len(self.chargers) >= 1:
            best_d2 = 1e18
            for (cx, cz) in self.chargers:
                d2 = (hx - cx) ** 2 + (hz - cz) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    self.last_charged_pos = (cx, cz)
            if self.last_charged_pos is not None:
                self.charger_visit_step[self.last_charged_pos] = self.step_no
            self.target_charger_pos = self._pick_lru_charger()

        if self.target_charger_pos is None and len(self.chargers) > 0:
            self.target_charger_pos = self._pick_lru_charger()

        # Target charger BFS: every 5 steps to save computation / 目标充电桩BFS：每5步算一次节省开销
        if self.step_no <= 1 or self.step_no % 5 == 0 or just_charged:
            self.bfs_d_target_charger, self.bfs_target_charger_dir = self._calc_bfs_to_target_charger()

        # Enhanced trapped detection: multi-dimensional / 增强被嚰検测：多维度判断
        pos_dist = float(np.sqrt((hx - self.last_escape_pos[0])**2 + (hz - self.last_escape_pos[1])**2))
        
        # 1. Position change detection / 位置変化检测
        position_change_low = pos_dist < 3.0
        
        # 2. Efficiency detection: steps vs new cells / 效率检测：步數 vs 新格子
        efficiency_low = False
        if self.step_no > 50:
            steps_per_new = self.step_no / max(len(self.visited), 1)
            efficiency_low = steps_per_new > 60  # 60 steps for only 1 new cell
        
        # 3. Movement speed detection / 速度检测：平均移动距离
        speed_low = False
        if len(self.recent_positions) > 20:
            recent_path = self.recent_positions[-20:]
            total_dist = sum([
                np.sqrt((recent_path[i][0]-recent_path[i-1][0])**2 + 
                       (recent_path[i][1]-recent_path[i-1][1])**2)
                for i in range(1, len(recent_path))
            ])
            avg_speed = total_dist / len(recent_path)
            speed_low = avg_speed < 1.5
        
        # Update stuck counter based on multi-dimensional detection / 根据多维度检测更新被嚰计数
        if position_change_low or speed_low:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_escape_pos = (hx, hz)
        
        # Enter escape mode: multiple conditions / 进入突囲模式：多条件
        should_escape = (self.stuck_counter >= 15 or efficiency_low)
        if should_escape and not self.escape_mode:
            self.escape_mode = True
            self.last_escape_pos = (hx, hz)
        
        # Exit escape mode / 退出突囲模式
        if self.escape_mode and pos_dist >= 10.0:
            self.escape_mode = False
            self.stuck_counter = 0

    def _calc_nearest_entity_dist(self, entities):
        """Nearest Euclidean distance to a set of (x,z) entities."""
        if not entities:
            return 200.0
        hx, hz = self.cur_pos
        dx = np.array([e[0] - hx for e in entities], dtype=np.float32)
        dz = np.array([e[1] - hz for e in entities], dtype=np.float32)
        d = np.sqrt(dx * dx + dz * dz)
        return float(np.min(d)) if d.size > 0 else 200.0

    def _bfs_distance(self, start, targets, max_expand=4096):
        """BFS shortest path distance on passable_map (4-neighborhood).

        Returns large value if unreachable or targets empty.
        """
        return self._bfs_distance_and_dir(start, targets, max_expand)[0]

    def _bfs_distance_and_dir(self, start, targets, max_expand=4096):
        """BFS shortest path distance AND first-step direction on passable_map.

        BFS最短路径距离及首步方向。返回 (distance, (dx, dz))。
        dx, dz 是从起点出发第一步的方向（归一化到{-1,0,1}），用于导航。
        """
        if not targets:
            return 256.0, (0, 0)
        sx, sz = int(start[0]), int(start[1])
        if (sx, sz) in targets:
            return 0.0, (0, 0)
        if not (0 <= sx < self.GRID_SIZE and 0 <= sz < self.GRID_SIZE):
            return 256.0, (0, 0)
        if self.passable_map[sx, sz] == 0:
            return 256.0, (0, 0)

        q = deque()
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        seen = set([(sx, sz)])
        # Each entry: (x, z, distance, first_step_dx, first_step_dz)
        for dx, dz in dirs:
            nx, nz = sx + dx, sz + dz
            if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                continue
            if self.passable_map[nx, nz] == 0:
                continue
            if (nx, nz) in targets:
                return 1.0, (dx, dz)
            seen.add((nx, nz))
            q.append((nx, nz, 1, dx, dz))

        expands = 0
        while q:
            x, z, d, fdx, fdz = q.popleft()
            expands += 1
            if expands > max_expand:
                break
            nd = d + 1
            for dx, dz in dirs:
                nx, nz = x + dx, z + dz
                if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                    continue
                if (nx, nz) in seen:
                    continue
                if self.passable_map[nx, nz] == 0:
                    continue
                if (nx, nz) in targets:
                    return float(nd), (fdx, fdz)
                seen.add((nx, nz))
                q.append((nx, nz, nd, fdx, fdz))
        return 256.0, (0, 0)

    def _pick_lru_charger(self):
        """Pick the Least Recently Used charger as next patrol target.

        选择最久未访问的充电桩作为下一个巡回目标。
        优先选从未访问的；同等时选离当前位置最远的（最大化覆盖面积）。
        """
        if not self.chargers:
            return None
        hx, hz = self.cur_pos
        best_pos = None
        best_key = (1e18, -1)  # (visit_step, -distance²) — minimize visit, maximize dist
        for (cx, cz) in self.chargers:
            visit = self.charger_visit_step.get((cx, cz), -1)  # -1 = never visited
            d2 = (cx - hx) ** 2 + (cz - hz) ** 2
            key = (visit, -d2)  # lower visit_step wins; tie-break: farther wins
            if key < best_key:
                best_key = key
                best_pos = (cx, cz)
        return best_pos

    def _calc_bfs_to_charger(self):
        """BFS to nearest charger, returns (distance, (first_step_dx, first_step_dz)).

        BFS到最近充电桩，返回（距离，首步方向）。
        使用较大的 max_expand 确保充电桩距离精确（关系生死）。
        """
        if not self.chargers:
            return 256.0, (0, 0)
        # organs size w=h=3, treat as 3x3 area around center / 充电桩占 3x3
        targets = set()
        for (x, z) in self.chargers:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    tx, tz = int(x + dx), int(z + dz)
                    if 0 <= tx < self.GRID_SIZE and 0 <= tz < self.GRID_SIZE:
                        targets.add((tx, tz))
        return self._bfs_distance_and_dir(self.cur_pos, targets, max_expand=8192)

    def _calc_bfs_to_target_charger(self):
        """BFS to the target charger (patrol destination), returns (distance, direction).

        BFS到目标充电桩（巡回目的地），返回（距离，首步方向）。
        """
        if self.target_charger_pos is None:
            return 256.0, (0, 0)
        tx, tz = self.target_charger_pos
        targets = set()
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nx, nz = int(tx + dx), int(tz + dz)
                if 0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE:
                    targets.add((nx, nz))
        return self._bfs_distance_and_dir(self.cur_pos, targets, max_expand=2048)

    def _calc_bfs_to_local_dirt(self):
        """BFS to nearest dirt, returns (distance, (first_step_dx, first_step_dz)).

        BFS到最近污渍，返回（距离，首步方向）。
        """
        view = self._view_map
        if view is None:
            return 256.0, (0, 0)
        dirt = np.argwhere(view == 2)
        if len(dirt) == 0:
            return 256.0, (0, 0)
        hx, hz = self.cur_pos
        half = self.VIEW_HALF
        targets = set()
        for ri, ci in dirt:
            gx = int(hx - half + int(ri))
            gz = int(hz - half + int(ci))
            if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                targets.add((gx, gz))
        return self._bfs_distance_and_dir(self.cur_pos, targets, max_expand=2048)

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (98D): dual-channel 3×3 avg pool on 21×21 view.

        局部视野特征（98D）：21×21视野按双通道做3×3平均池化，输出7×7×2=98D。
        通道: [0]障碍率 [1]污渍率。
        已清扫率 = 1 - 障碍率 - 污渍率，线性冗余故省略。
        """
        view = self._view_map  # values: 0=obstacle, 1=cleaned, 2=dirt
        h, w = view.shape
        ph, pw = h // 3, w // 3
        blocks = view[:ph*3, :pw*3].reshape(ph, 3, pw, 3)  # (7, 3, 7, 3)
        obstacle = (blocks == 0).mean(axis=(1, 3)).astype(np.float32)  # (7,7)
        dirt     = (blocks == 2).mean(axis=(1, 3)).astype(np.float32)  # (7,7)
        return np.concatenate([obstacle.flatten(), dirt.flatten()])  # 98D

    def _get_global_state_feature(self):
        """Global state feature (9D).

        全局状态特征（9D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  wall_hit          wall collision last step / 上步是否撞墙 {0,1}
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  battery_max_norm  total capacity context / 电池总容量上下文 [0,1]
          [7]  n_chargers_norm   charger count context / 充电桩数量上下文 [0,1]
          [8]  n_npcs_norm       NPC count context / NPC数量上下文 [0,1]
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # Generalization context: let agent "see" the current config
        # 泛化上下文：让智能体"看到"当前配置，以便多配置训练时学会不同策略
        # 用配置值而非帧级可见列表长度，确保特征稳定不波动
        battery_max_norm = float(np.clip(self.battery_max / 300.0, 0.0, 1.0))
        n_chargers_norm = float(np.clip(self.config_charger_count / 4.0, 0.0, 1.0))
        n_npcs_norm = float(np.clip(self.config_npc_count / 4.0, 0.0, 1.0))

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                float(self.wall_hit),
                pos_x_norm,
                pos_z_norm,
                battery_max_norm,
                n_chargers_norm,
                n_npcs_norm,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _compute_true_legal_action(self):
        """Compute actual passability for 8 directions from local view map.

        根据局部视野地图计算8个方向的实际可通行性（含斜向防穿角规则）。
        环境返回的 legal_act 始终全1，此方法提供真实的可通行掩码，防止智能体撞墙浪费步数和电量。
        """
        view = self._view_map
        if view is None or view.shape[0] < 21:
            return [1] * 8

        cx, cz = self.VIEW_HALF, self.VIEW_HALF  # center (10, 10) = agent
        # Action -> (dx, dz): 0=右 1=右上 2=上 3=左上 4=左 5=左下 6=下 7=右下
        dirs = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        legal = []
        for dx, dz in dirs:
            tx, tz = cx + dx, cz + dz
            # Target cell must be within view and passable (≠0)
            if not (0 <= tx < 21 and 0 <= tz < 21) or view[tx, tz] == 0:
                legal.append(0)
                continue
            # Diagonal: anti-corner-cutting (至少一个正交邻居可通行)
            if abs(dx) == 1 and abs(dz) == 1:
                h_pass = view[cx + dx, cz] != 0  # horizontal neighbor
                v_pass = view[cx, cz + dz] != 0  # vertical neighbor
                if not h_pass and not v_pass:
                    legal.append(0)
                    continue
            legal.append(1)

        # Fallback: if all blocked, allow all to avoid softmax NaN / 全封时允许全部
        if sum(legal) == 0:
            return [1] * 8
        return legal

    def get_legal_action(self):
        """Return true legal action mask (8D list) computed from local view.

        返回基于视野地图计算的真实合法动作掩码（8D list）。
        不可通行方向将被策略网络自动屏蔽，智能体永远不会选择撞墙动作。
        """
        return self._compute_true_legal_action()

    def _action_hist_onehot(self):
        """One-hot encode last K actions (K*8). Invalid (-1) -> all zeros."""
        k = self.action_hist_k
        out = np.zeros((k, 8), dtype=np.float32)
        for i, a in enumerate(self.action_hist[-k:]):
            if 0 <= int(a) < 8:
                out[i, int(a)] = 1.0
        return out.flatten()

    def _charger_feats(self):
        """Charger features (7D): nearest(dist,dx,dz) + target(dist,dx,dz) + urgency.

        充电桩特征（7D）：最近充电桩(距离,方向dx,dz) + 目标充电桩(距离,方向dx,dz) + 紧迫度。
        最近充电桩 = 安全保障（低电量往这走）
        目标充电桩 = 巡回方向（充完电往那走）
        """
        max_bfs = 256.0
        # Nearest charger (for safety) / 最近充电桩（安全保障）
        d_near = float(np.clip(self.bfs_d_charger / max_bfs, 0.0, 1.0))
        near_dx = float(self.bfs_charger_dir[0])
        near_dz = float(self.bfs_charger_dir[1])

        # Target/next charger (for patrol) / 目标充电桩（巡回方向）
        d_target = float(np.clip(self.bfs_d_target_charger / max_bfs, 0.0, 1.0))
        target_dx = float(self.bfs_target_charger_dir[0])
        target_dz = float(self.bfs_target_charger_dir[1])

        # Urgency: battery_max independent
        # 充电紧迫度
        reach_r = self.battery / max(self.bfs_d_charger, 1.0)
        charge_urgency = float(np.clip((3.0 - reach_r) / 2.0, 0.0, 1.0))
        return np.array([d_near, near_dx, near_dz, d_target, target_dx, target_dz, charge_urgency], dtype=np.float32)

    def _npc_feats(self):
        max_d = 50.0
        hx, hz = self.cur_pos
        safe_radius = 2.5
        if not self.npcs:
            d = max_d
            dx = 0.0
            dz = 0.0
        else:
            best = None
            best_d2 = 1e18
            for (x, z) in self.npcs:
                ddx, ddz = float(x - hx), float(z - hz)
                d2 = ddx * ddx + ddz * ddz
                if d2 < best_d2:
                    best_d2 = d2
                    best = (ddx, ddz)
            dx, dz = best if best is not None else (0.0, 0.0)
            d = float(np.sqrt(best_d2)) if best is not None else max_d

        d_norm = _norm(d, max_d)
        dx_norm = float(np.clip(dx / max_d, -1.0, 1.0))
        dz_norm = float(np.clip(dz / max_d, -1.0, 1.0))
        too_close = 1.0 if d < safe_radius else 0.0
        return np.array([d_norm, dx_norm, dz_norm, too_close], dtype=np.float32)

    def _traj_feats(self):
        # K*8 action hist onehot + last move dx,dz + loop flag
        hx, hz = self.cur_pos
        lx, lz = self.last_pos
        dx = float(hx - lx)
        dz = float(hz - lz)
        max_step = 3.0  # per-step move roughly within 1-3 cells
        dx_norm = float(np.clip(dx / max_step, -1.0, 1.0))
        dz_norm = float(np.clip(dz / max_step, -1.0, 1.0))
        return np.concatenate(
            [
                self._action_hist_onehot(),
                np.array([dx_norm, dz_norm, float(self.loop_flag)], dtype=np.float32),
            ]
        )

    def _memory_feats(self):
        """Memory feature (5D): coverage, recent_coverage, cleaned_norm, steps_since_clean_norm, idle_norm.

        记忆特征（5D）：覆盖率、最近覆盖率、清扫格数、距上次清扫步数、最近50步空闲比率。
        """
        cleaned_cnt = float(len(self.step_cleaned_cells))
        cleaned_norm = float(np.clip(cleaned_cnt / 10.0, 0.0, 1.0))
        # 距上次成功清扫的步数（归一化，cap=50）— 与 battery_max 无关
        steps_since_clean_norm = float(np.clip(self.steps_since_clean / 50.0, 0.0, 1.0))
        # 最近50步空闲比率（滑动窗口）
        idle_norm = float(sum(self.idle_window)) / max(len(self.idle_window), 1)
        return np.array(
            [
                float(np.clip(self.coverage_ratio, 0.0, 1.0)),
                float(np.clip(self.recent_coverage, 0.0, 1.0)),
                cleaned_norm,
                steps_since_clean_norm,
                idle_norm,
            ],
            dtype=np.float32,
        )

    def _bfs_feats(self):
        """BFS features (6D): charger_dist, dirt_dist, dirt_dx, dirt_dz, reach_ratio, urgency.

        BFS特征（6D）：充电桩距离、污渍距离、污渍首步方向dx/dz、电量/距离比、充电紧迫度。
        dirt_dx/dz 告诉智能体"走哪个方向能最快到达最近污渍"——这是减少空步、提升分数的关键信号。
        reach_ratio 和 urgency 完全不依赖 battery_max。
        """
        max_bfs = 256.0
        d_ch = float(np.clip(self.bfs_d_charger, 0.0, max_bfs))
        d_di = float(np.clip(self.bfs_d_dirt, 0.0, max_bfs))
        # BFS first-step direction toward nearest dirt (values in {-1, 0, 1})
        # BFS到最近污渍的首步方向
        dirt_dx = float(self.bfs_dirt_dir[0])
        dirt_dz = float(self.bfs_dirt_dir[1])
        # reach_ratio: how many trips to charger the battery can afford (battery_max independent)
        # 电量够跑几趟充电桩（与 battery_max 完全无关）
        reach_ratio = float(np.clip(
            self.battery / max(self.bfs_d_charger, 1.0), 0.0, 5.0
        )) / 5.0  # normalized to [0, 1]
        # urgency: binary signal — must head to charger NOW (battery_max independent)
        # 二值紧迫信号 — 电量仅够1.5倍距离时触发（与 battery_max 完全无关）
        urgency = 1.0 if self.battery <= self.bfs_d_charger * 1.5 else 0.0
        return np.array([d_ch / max_bfs, d_di / max_bfs, dirt_dx, dirt_dz, reach_ratio, urgency], dtype=np.float32)

    def feature_process(self, env_obs, last_action):
        """Generate 148D feature vector, legal action mask, and scalar reward.

        生成 148D 特征向量（98D视野 + 9D全局 + 8D合法动作 + 7D充电桩 + 4D NPC + 11D轨迹 + 5D记忆 + 6D BFS）、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 98D (obstacle+dirt each 7x7)
        global_state = self._get_global_state_feature()  # 9D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        # Phase1/2 extra features / 新增特征
        charger_feats = self._charger_feats()  # 7D (nearest+target+urgency)
        npc_feats = self._npc_feats()  # 4D
        traj_feats = self._traj_feats()  # 11D (K=1 => 8 + 3)
        memory_feats = self._memory_feats()  # 5D
        bfs_feats = self._bfs_feats()  # 6D (dist+dir+reach+urgency)

        feature = np.concatenate(
            [local_view, global_state, legal_arr, charger_feats, npc_feats, traj_feats, memory_feats, bfs_feats]
        )

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        # Weights (清扫优先模式) / 权重（清扫为主要目标，充电低电量门控）
        w_clean = 0.10       # 清扫奖励：主要目标
        # 充电奖励：动态权重 = 0.05 + 0.20*urgency（见下方）
        w_npc = 0.03         # NPC躲避：保持不变
        w_new = 0.002        # 探索新格子：适当提升（原0.0005）
        w_repeat = 0.0005    # 回环惩罚：适当提升（原0.0002）
        w_streak = 0.001     # 连清奖励：适当提升（原0.0005）
        w_idle = 0.001       # 空跑惩罚：适当提升（原0.0005）

        # Cleaning reward / 清扫奖励
        self.cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        if self.cleaned_this_step > 0:
            self.clean_streak += 1
            self.idle_steps = 0
        else:
            self.clean_streak = 0
            self.idle_steps += 1

        r_clean = w_clean * float(self.cleaned_this_step)

        streak_cap = 20
        r_streak = w_streak * float(min(self.clean_streak, streak_cap))
        r_idle = -w_idle if self.cleaned_this_step == 0 else 0.0

        # Explore: new cell reward + repeat penalty / 探索：新格子 + 回环惩罚
        r_new = w_new if self.is_new_cell else 0.0
        r_repeat = -w_repeat * float(self.loop_flag)

        # Charging: urgency-scaled reward (battery_max independent)
        # 充电：紧迫度缩放奖励（与 battery_max 无关）
        reach_r = self.battery / max(self.bfs_d_charger, 1.0)
        urgency = float(np.clip((2.0 - reach_r) / 2.0, 0.0, 1.0))

        # Component 1: Battery danger penalty (creates VALUE pressure)
        # 低电量危险惩罚（创造价值压力，让 critic 学到"低电量=危险"）
        # urgency=0 → 0, urgency=0.5 → -0.05/step, urgency=1.0 → -0.10/step
        r_battery_danger = -0.10 * urgency

        # Component 2: Charging approach reward using RAW BFS delta
        # 充电接近奖励：用原始BFS距离差（每步±1），非归一化的（/256后太小）
        # 权重随紧迫度递增：urgency=0 → 0.05, urgency=1 → 0.25（超过 r_clean=0.10）
        if self.step_no <= 1:
            self._prev_bfs_d_charger = self.bfs_d_charger
        bfs_delta = self._prev_bfs_d_charger - self.bfs_d_charger  # >0 = getting closer
        bfs_delta = float(np.clip(bfs_delta, -2.0, 2.0))  # cap to avoid spikes
        w_charge_dynamic = 0.05 + 0.20 * urgency
        r_charge_approach = w_charge_dynamic * bfs_delta
        self._prev_bfs_d_charger = self.bfs_d_charger

        # Component 3: Actual charging reward (battery increased this step)
        # 实际充电奖励：踏上充电桩充电时，按电量缺失比例给奖
        # 电量越低奖励越大，电量满后自然趋零 → 智能体不会赖着不走
        r_charge_actual = 0.0
        if self.just_charged:
            battery_deficit = 1.0 - (self.last_battery / max(self.battery_max, 1))
            r_charge_actual = 0.15 * battery_deficit

        # Component 4: Near-charger low-battery proximity bonus
        # 近桩低电量引力：离桩 ≤3步 且 电量<70% 时额外激励靠近
        # 用绝对电量比（非reach_r），解决"近桩时urgency=0"的盲区
        r_proximity = 0.0
        battery_ratio_raw = self.battery / max(self.battery_max, 1)
        if self.bfs_d_charger <= 3 and battery_ratio_raw < 0.7:
            proximity_scale = 1.0 - self.bfs_d_charger / 4.0   # BFS=0→1.0, 1→0.75, 2→0.5, 3→0.25
            deficit_scale = (0.7 - battery_ratio_raw) / 0.7     # 0%→1.0, 35%→0.5, 70%→0.0
            r_proximity = 0.06 * proximity_scale * deficit_scale

        r_charge = r_battery_danger + r_charge_approach + r_charge_actual + r_proximity

        # Step penalty: constant small negative to encourage efficiency
        # 步数惩罚：固定小负值鼓励高效清扫，避免分段逻辑干扰策略学习
        step_penalty = -0.001

        # NPC avoid: soft penalty inside safe radius / NPC 躲避：安全半径内软惩罚
        safe_radius = 4.0
        d_npc = float(self.d_npc_min)
        s = max(0.0, (safe_radius - d_npc) / safe_radius)
        r_npc = -w_npc * (s * s)

        # Extra severe penalty when extremely close to NPC / 极近距离额外惩罚
        if d_npc < 1.0:
            r_npc = min(r_npc, -0.10)

        # Escape mode: reduce NPC penalty and add escape reward / 突围模式：降低NPC惩罚并添加突围奖励
        r_escape = 0.0
        if self.escape_mode:
            # In escape mode, allow crossing NPC to get out / 突围模式下允许穿越NPC
            r_npc = r_npc * 0.3
            # Reward for moving away from stuck position / 奖励远离被困位置
            pos_dist = float(np.sqrt((self.cur_pos[0] - self.last_escape_pos[0])**2 + (self.cur_pos[1] - self.last_escape_pos[1])**2))
            r_escape = 0.01 * min(pos_dist / 10.0, 1.0)  # reward proportional to escape progress

        # Wall collision penalty: wasted step + battery / 撞墙惩罚：浪费步数+电量
        wall_penalty = -0.02 * self.wall_hit

        # Dirt approach reward: incentivize moving toward nearest dirt
        # 污渍接近奖励：激励移向最近污渍，减少空步
        if self.step_no <= 1:
            self._prev_bfs_d_dirt = self.bfs_d_dirt
        dirt_bfs_delta = self._prev_bfs_d_dirt - self.bfs_d_dirt  # >0 = closer to dirt
        dirt_bfs_delta = float(np.clip(dirt_bfs_delta, -2.0, 2.0))
        r_dirt_approach = 0.03 * dirt_bfs_delta if self.cleaned_this_step == 0 else 0.0
        self._prev_bfs_d_dirt = self.bfs_d_dirt

        total = float(r_clean + r_charge + r_npc + r_new + r_repeat + r_streak + r_idle + r_escape + step_penalty + wall_penalty + r_dirt_approach)

        self.last_reward_components = {
            "cleaning": float(r_clean),
            "bat_danger": float(r_battery_danger),
            "charge_approach": float(r_charge_approach),
            "charge_actual": float(r_charge_actual),
            "proximity": float(r_proximity),
            "npc_avoid": float(r_npc),
            "explore_new": float(r_new),
            "explore_repeat": float(r_repeat),
            "eff_streak": float(r_streak),
            "eff_idle": float(r_idle),
            "escape": float(r_escape),
            "step_penalty": float(step_penalty),
            "wall_penalty": float(wall_penalty),
            "dirt_approach": float(r_dirt_approach),
            "urgency": float(urgency),
            "total": total,
        }

        return total
