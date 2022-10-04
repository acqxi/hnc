import copy
import glob
import os
import random as rd
import re
from operator import xor
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.transform as sk_xfmr
import torch
import torchvision.transforms as xfmr
from scipy.ndimage import rotate as sci_rotate
from torch.utils.data import DataLoader, Dataset


class LAPsGetter:
    def __init__(self, paths, isShuffle: bool = True):
        self.paths = paths
        self.refAry = np.array([np.load(p) for p in self.paths])
        self.load_times = 0
        self.shuffle = isShuffle
        self._reload(isShuffle)

    def __len__(self):
        return len(self.paths)

    def _reload(self, shuffle: bool = True):
        self.load_times += 1
        self.indices = list(range(len(self.paths)))
        if shuffle:
            rd.shuffle(self.indices)

    def get(self):
        if len(self.indices) == 0:
            self._reload(self.shuffle)
        targ = self.indices.pop()
        return self.refAry[targ].copy(), self.paths[targ]

    def gets(self, number: int) -> np.ndarray:
        ref_idxs = self.indices[-min(len(self.indices), number) :]
        while number != len(ref_idxs):
            self._reload(self.shuffle)
            ref_idxs.extend(self.indices[-min(len(self.indices), number - len(ref_idxs)) :])
        return self.refAry[ref_idxs], self.paths[ref_idxs]

    def reset(self):
        self.load_times = 0
        self._reload(self.shuffle)
        
        
def data_split_patient(paths, ratio: List[int] = [7, 1, 2], debug=False) -> tuple:
    sets = {"trains": [], "valids": [], "tests": []}
    names = list(sets.keys())
    ratio_n = {names[i]: ratio[i] / sum(ratio) for i in range(len(ratio))}
    last_patient, last_choice = 0, -1
    for p in sorted(paths, key=lambda x: int(re.findall(r"(\d{3})_20", x)[0])):
        patient = re.findall(r"(\d{3})_20", p)[0]
        if last_patient != patient:
            names = [n for n in names if len(sets[n]) < len(paths) * ratio_n[n]]
            last_choice = rd.choices(names, weights=[ratio_n[n] for n in names])[0]
            last_patient = patient
        sets[last_choice].append(p)
    if debug:
        print(f"train:valid:test = {':'.join(map(str, ratio))} = {':'.join([f'{len(v)}' for v in sets.values()])}")

    return sets


def data_split_node(paths, ratio: List[int] = [7, 1, 2], type: List[int] = [0, 1, 2], debug=False) -> tuple:
    sets = {"trains": [], "valids": [], "tests": []}
    names = list(sets.keys())
    ratio_n = {names[i]: ratio[i] / sum(ratio) for i in range(len(ratio))}
    for p in paths:
        names = [n for n in names if len(sets[n]) < len(paths) * ratio_n[n]]
        sets[rd.choices(names, weights=[ratio_n[n] for n in names])[0]].append(p)
    if debug:
        print(f"train:valid:test = {':'.join(map(str, ratio))} = {':'.join([f'{len(v)}' for v in sets.values()])}")

    return sets


def show_info(dataset, anno: Optional[str] = None) -> None:
    anno = anno or " "
    tmps = []
    print(f"{anno:{max(len(anno), 7)}} {'LNM-':^9} {'LNM+ENE-':^9} {'LNM+ENE+':^9} {'inPhase':^9}", end="")
    for k, v in dataset.items():
        print(f"\n{k:{max(len(anno), 7)}}", end=" ")
        tmp = [re.findall(r"_(\d)\.", x)[0] for x in v]
        tmps += tmp
        for i in range(3):
            print(f"{tmp.count(str(i)):^9}", end=" ")
        print(f"{len(tmp):^9}", end=" ")

    print(f"\n{'total':{max(len(anno), 7)}}", end=" ")
    for i in range(3):
        print(f"{tmps.count(str(i)):^9}", end=" ")
    print(f"{len(tmps):^9}")


class LAPsDatasetNode:
    def __init__(
        self,
        dataRoot="./data/laps/ex5/size-preserved",
        mode: str = ["LNM", "ENE"][0],
        splitRatio: List[int] = [7, 1, 2],
        balanced: bool = True,
        isShuffle: bool = True,
        xfmr: Optional[xfmr.Compose] = None,
    ):
        self.root = dataRoot
        self.mode = mode
        self.ratio = splitRatio
        self.balanced = balanced
        self.shuffle = isShuffle
        self.xfmr = xfmr
        self.phase = None

        self.paths = [p for p in glob.glob(os.path.join(dataRoot, "*.npy"))]
        self.get_name = {'lap':lambda p : os.path.basename(p), 'patient': lambda p:re.findall(r"(\d{3})_20", p)[0]}

        print(f"read data folder : {dataRoot}")
        print(f"load {len(self.paths)} Lymph Node images")

        self.split_dataset(self.ratio)
        self.show_split_info()

        self.which_in_set()

    @property
    def copy(self):
        return copy.copy(self)

    def split_dataset(self, ratio):
        print(f"train:valid:test = {':'.join(map(str, ratio))}")
        self.raw_sets = tmp = [
            data_split_node([p for p in self.paths if p.endswith(f"{i}.npy")]) for i in range(3)
        ]  # node
        self.sets = {
            "LNM": {k: (tmp[0][k] + tmp[1][k] + tmp[2][k]) for k in tmp[0]},
            "ENE": {k: (tmp[1][k] + tmp[2][k]) for k in tmp[0]},
        }
        self.set_loader()

    def show_split_info(self, k: Optional[str] = None) -> None:
        print()
        if k in self.sets:
            show_info(self.sets[k], anno=k + "sets")
            return
        for k in self.sets.keys():
            show_info(self.sets[k], anno=k + "sets")
            print()
            
    def _in_set(self, mode:str=['lap', 'patient'][0], filePath: Optional[str] = None):
        for k in ["valids", "tests"]:
            tmp_names = set([self.get_name[mode](p) for p in self.sets["LNM"][k]])
            if filePath:
                os.makedirs(os.path.dirname(filePath), exist_ok=True)
                with open(filePath, "a") as f:
                    f.writelines((f"LAPs in set {k:6}", ":", ", ".join(sorted(list(tmp_names)))))
                    f.writelines("\n\n\n")
            else:
                print(f"{mode.upper()}s in set {k:6}", ":", ", ".join(sorted(list(tmp_names))))

    def who_in_set(self, mode=1, filePath: Optional[str] = None):
        self._in_set('patient', filePath)

    def which_in_set(self, mode=1, filePath: Optional[str] = None):
        self._in_set('lap', filePath)
                
    def _load_split(self, valid: str, test: str, mode:str=['lap', 'patient'][0]):
        valids = valid.split(", ")
        tests = test.split(", ")
        self.sets = {"LNM": {"trains": [], "valids": [], "tests": []}, "ENE": {"trains": [], "valids": [], "tests": []}}
        self.raw_sets = [{"trains": [], "valids": [], "tests": []} for i in range(3)]
        for p in self.paths:
            name = self.get_name[mode](p)
                    
            if name in valids:
                which = "valids"
            elif name in tests:
                which = "tests"
            else:
                which = "trains"

            if p.endswith(f"{0}.npy"):
                self.sets["ENE"][which].append(p)
            self.sets["LNM"][which].append(p)
            for i in range(3):
                if p.endswith(f"{i}.npy"):
                    self.raw_sets[i][which].append(p)
        self.set_loader()
        

    def load_split_patient(self, valid: str, test: str):
        self._load_split(valid, test, 'patient')

    def load_split_node(self, valid: str, test: str):
        self._load_split(valid, test, 'lap')

    def set_loader(self):
        tmp = self.raw_sets
        self.gets = {
            "LNM": {
                k: [LAPsGetter(ps) for ps in [tmp[0][k], tmp[1][k] + tmp[2][k], tmp[0][k] + tmp[1][k] + tmp[2][k]]]
                for k in tmp[0]
            },  # 0, 1, all
            "ENE": {k: [LAPsGetter(ps) for ps in [tmp[1][k], tmp[2][k], tmp[1][k] + tmp[2][k]]] for k in tmp[0]},
        }
        self.length = {
            mode: {phase: [max(len(pv[0]), len(pv[1])) * 2, len(pv[2])] for phase, pv in mode_v.items()}
            for mode, mode_v in self.gets.items()
        }

        if self.balanced:
            print(
                f"balanced to 1:1, training Lymph Nodes from {self.length[self.mode]['trains'][1]} to {self.length[self.mode]['trains'][0]}"
            )

    def set_phase(self, phase):
        self.phase = phase
        return self.copy

    def set_mode(self, mode):
        self.mode = mode
        self.show_split_info(mode)
        return self.copy

    def __len__(self):
        if self.phase is None:
            raise ValueError("have to set phase before use")
        return self.length[self.mode][self.phase][0 if self.balanced else 1]

    def __getitem__(self, idx):
        if self.phase is None:
            raise ValueError("have to set phase before use")

        if self.balanced:
            img, path = self.gets[self.mode][self.phase][idx % 2].get()
            label = idx % 2
        else:
            img, path = self.gets[self.mode][self.phase][2].get()
            label = int(re.findall(r"(\d)\.", path)[0])
            if self.mode == "LNM" and label == 2:
                label = 1
            elif self.mode == "ENE":
                label -= 1
        img = np.expand_dims(img, axis=0)
        if self.xfmr is not None:
            img: torch.Tensor = self.xfmr(torch.from_numpy(img).float())
        return img.float().contiguous(), label, path


class LAPsDatasetNodes(LAPsDatasetNode):
    def __init__(
        self,
        scaledDataRoot="./data/laps/ex5/size-scaled",
        mode: str = ["LNM", "ENE"][0],
        splitRatio: List[int] = [7, 1, 2],
        balanced: bool = True,
        isShuffle: bool = True,
        xfmr: Optional[xfmr.Compose] = None,
    ):
        super().__init__(scaledDataRoot, mode, splitRatio, balanced, isShuffle, xfmr)
        print(f"read data folder : {scaledDataRoot.replace('size-scaled','size-preserved')}")

    def _copy_split_to_preserved(self):
        self.sets2 = {}
        self.raw_sets2 = [{}, {}, {}]
        for mode in self.sets:
            self.sets2[mode] = {}
            for phase in self.sets[mode]:
                self.sets2[mode][phase] = []
                for p_or_ps in self.sets[mode][phase]:
                    self.sets2[mode][phase].append(p_or_ps.replace("size-scaled", "size-preserved"))
                for i in range(3):
                    self.raw_sets2[i][phase] = [p for p in self.sets2["LNM"][phase] if p.endswith(f"{i}.npy")]
        tmp = self.raw_sets2
        self.gets2 = {
            "LNM": {
                k: [LAPsGetter(ps) for ps in [tmp[0][k], tmp[1][k] + tmp[2][k], tmp[0][k] + tmp[1][k] + tmp[2][k]]]
                for k in tmp[0]
            },  # 0, 1, all
            "ENE": {k: [LAPsGetter(ps) for ps in [tmp[1][k], tmp[2][k], tmp[1][k] + tmp[2][k]]] for k in tmp[0]},
        }

    def set_loader(self):
        super().set_loader()
        self._copy_split_to_preserved()

    def __getitem__(self, idx):
        if self.phase is None:
            raise ValueError("have to set phase before use")

        if self.balanced:
            img, path = self.gets[self.mode][self.phase][idx % 2].get()
            img2, _ = self.gets2[self.mode][self.phase][idx % 2].get()
            label = idx % 2
        else:
            img, path = self.gets[self.mode][self.phase][2].get()
            img2, _ = self.gets2[self.mode][self.phase][2].get()
            label = int(re.findall(r"(\d)\.", path)[0])
            if self.mode == "LNM" and label == 2:
                label = 1
            elif self.mode == "ENE":
                label -= 1
        img = np.expand_dims(img, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        if self.xfmr is not None:
            img: torch.Tensor = self.xfmr(torch.from_numpy(img).float())
            img2: torch.Tensor = self.xfmr(torch.from_numpy(img2).float())
        else:
            img: torch.Tensor = torch.from_numpy(img).float()
            img2: torch.Tensor = torch.from_numpy(img2).float()
            
        return img.float().contiguous(), img2.float().contiguous(), label, path
