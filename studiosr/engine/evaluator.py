import os
from typing import Callable, List, Tuple

import cv2
import numpy as np

from studiosr.data import PairedImageDataset, GalaxyPairedImageDataset, Test_GalaxyPairedImageDataset
from studiosr.utils import compare, compute_psnr, compute_ssim, gdown_and_extract, ours_compare


class Evaluator:
    """
    A class for evaluating the performance of super-resolution models on image datasets.

    Args:
        dataset (str, optional): The name of the evaluation dataset (default is "DIV2K_mini").
        scale (int, optional): The scaling factor for super-resolution (default is 4).
        root (str, optional): The root directory where evaluation dataset is located (default is "data").

    Note:
        This class is designed for evaluating the performance of super-resolution models. It loads the
        evaluation dataset, calculates PSNR and SSIM values for the model's output, and optionally visualizes
        the results. The class can be used for various evaluation datasets and scaling factors.
    """

    def __init__(
        self,
        dataset: str = "DIV2K_mini",
        scale: int = 4,
        root: str = "dataset",
    ) -> None:
        self.dataset = dataset
        self.scale = scale
        self.root = root
        root = self.download_dataset(self.root, self.dataset)
        gt_mod = 12 if scale in [2, 3, 4] else scale
        gt_path = os.path.join(root, f"GTmod{gt_mod}")
        lq_path = os.path.join(root, f"LRbicx{scale}")
        self.scale = scale
        self.testset = PairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        psnr, ssim = self.run(func, y_only, visualize, logging)
        print(f" {self.dataset:>8} - Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
        return psnr, ssim

    def run(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = False,
    ) -> Tuple[float, float]:
        crop_border = self.scale
        psnrs, ssims = [], []
        for i, (lq, gt) in enumerate(self.testset):
            sr = func(lq)
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)
            if logging:
                print(
                    f" {self.dataset:>8} - {i + 1:>3}/{len(self.testset):>3} PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}",
                    end="\r",
                )
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                compare([nn[:, :, ::-1], bc[:, :, ::-1], sr[:, :, ::-1], gt[:, :, ::-1]])
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        return psnr, ssim

    @staticmethod
    def download_dataset(root: str = "dataset", dataset: str = "Set5") -> str:
        dataset_id = {
            "Set5": "18bimJIcXV0nxYU9y64Liwo63afEZXlAY",
            "Set14": "1Wn8mJRFT7N4z0cGbqwGev4ltbLwi4Sg2",
            "BSD100": "1qoiBkwiUgv62MISQh4A4nibdmDfP5qzJ",
            "Urban100": "1YTYp0gVJj2gpIsL3N8NkEDKEPIZeyhnf",
            "Manga109": "1ZaUD3ZeaaI3zHlEI6HRSx0baBU2CeYe7",
            "DIV2K": "1kUlppta5vEmXa76EHU_mb6_EoibNWlXw",
            "DIV2K_mini": "1pDEDDuYzaRzmJb6ztZTafeui1xE6iCz9",
        }
        benchmark_path = os.path.join(root, dataset)
        if not os.path.exists(benchmark_path):
            os.makedirs(root, exist_ok=True)
            id = dataset_id[dataset]
            gdown_and_extract(id=id, save_dir=root)
        return benchmark_path

    @staticmethod
    def benchmark(
        func: Callable[[np.ndarray], np.ndarray],
        scale: int = 4,
        y_only: bool = True,
        datasets: List[str] = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"],
    ) -> Tuple[List[float], List[float]]:
        log_data = "| Metric |"
        log_line = "| ------ |"
        log_psnr = "|   PSNR |"
        log_ssim = "|   SSIM |"

        psnr_list, ssim_list = [], []
        for dataset in datasets:
            psnr, ssim = Evaluator(dataset, scale).run(func, y_only, logging=True)
            log_data += " %10s |" % dataset
            log_line += " ---------- |"
            log_psnr += " %10.3f |" % psnr
            log_ssim += " %10.4f |" % ssim
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        print(log_data)
        print(log_line)
        print(log_psnr)
        print(log_ssim)
        print()

        return psnr_list, ssim_list


class Evaluator2:
    def __init__(
        self,
        dataset: str = "Set5",
        scale: int = 4,
        root: str = "dataset/benchmark",
    ) -> None:
        self.dataset = dataset
        self.scale = scale
        self.root = root
        root = self.download_dataset(self.root, self.dataset)
        gt_path = os.path.join(root, "HR")
        lq_path = os.path.join(root, "LR_bicubic", f"X{scale}")
        self.scale = scale
        self.testset = PairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        psnr, ssim = self.run(func, y_only, visualize, logging)
        print(f" {self.dataset:>8} - Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
        return psnr, ssim

    def run(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = False,
    ) -> Tuple[float, float]:
        crop_border = self.scale
        psnrs, ssims = [], []
        for i, (lq, gt) in enumerate(self.testset):
            sr = func(lq)
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)
            if logging:
                print(
                    f" {self.dataset:>8} - {i + 1:>3}/{len(self.testset):>3} PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}",
                    end="\r",
                )
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                compare([nn[:, :, ::-1], bc[:, :, ::-1], sr[:, :, ::-1], gt[:, :, ::-1]])
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        return psnr, ssim

    @staticmethod
    def download_dataset(root: str = "dataset/benchmark", dataset: str = "Set5") -> str:
        dataset_id = {
            "Set5": "1ewFsDc-FdxierrNv8bGp4tE1BJzccyyr",
            "Set14": "1r_G-bFrjt-1puTJTMAxeLaI-fyiqlHN_",
            "BSD100": "1JAqwq03cu73HImotXxudstGPSyXB74eA",
            "Urban100": "1srG5FmDmnogUzvOywH7i2QfUnLsNGmxb",
        }
        benchmark_path = os.path.join(root, dataset)
        if not os.path.exists(benchmark_path):
            os.makedirs(root, exist_ok=True)
            id = dataset_id[dataset]
            gdown_and_extract(id=id, save_dir=root)
        return benchmark_path


def benchmark(
    func: Callable[[np.ndarray], np.ndarray],
    scale: int = 4,
    y_only: bool = True,
    datasets: List[str] = ["Set5", "Set14", "BSD100", "Urban100"],
) -> Tuple[List[float], List[float]]:
    log_data = "| Metric |"
    log_line = "| ------ |"
    log_psnr = "|   PSNR |"
    log_ssim = "|   SSIM |"

    psnr_list, ssim_list = [], []
    for dataset in datasets:
        psnr, ssim = Evaluator2(dataset, scale).run(func, y_only, logging=True)
        log_data += " %10s |" % dataset
        log_line += " ---------- |"
        log_psnr += " %10.3f |" % psnr
        log_ssim += " %10.4f |" % ssim
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(log_data)
    print(log_line)
    print(log_psnr)
    print(log_ssim)
    print()

    return psnr_list, ssim_list

class Evaluator_Galaxy64128:
    def __init__(
        self,
        dataset: str = "Galaxy",
        scale: int = 2,
        root: str = "/data4/GalaxySynthesis/Galaxy_SR_Dataset/gt_128_lq_64/minmax/minmax_merge_ttv/val",
    ) -> None:
        self.dataset = dataset
        self.scale = scale
        self.root = root
        # root = self.download_dataset(self.root, self.dataset)
        gt_path = os.path.join(root, "gt_minmax")
        lq_path = os.path.join(root, "lr_minmax")
        self.testset = GalaxyPairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        psnr, ssim = self.run(func, y_only, visualize, logging)
        print(f" {self.dataset:>8} - Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
        return psnr, ssim

    def run(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        crop_border = self.scale
        psnrs, ssims = [], []
        for i, (lq, gt) in enumerate(self.testset):
            sr = func(lq)
            sr = np.squeeze(sr)
            gt = np.squeeze(gt)
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)
            if logging:
                print(
                    f" {self.dataset:>8} - {i + 1:>3}/{len(self.testset):>3} PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}",
                    end="\r",
                )
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                compare([nn[:, :, ::-1], bc[:, :, ::-1], sr[:, :, ::-1], gt[:, :, ::-1]])
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        return psnr, ssim


class Evaluator_Galaxy3264:
    def __init__(
        self,
        dataset: str = "Galaxy",
        scale: int = 2,
        root: str = "/data4/GalaxySynthesis/Galaxy_SR_Dataset/gt_64_lq_32/minmax/minmax_merge_ttv/val",
    ) -> None:
        self.dataset = dataset
        self.scale = scale
        self.root = root
        # root = self.download_dataset(self.root, self.dataset)
        gt_path = os.path.join(root, "gt_minmax")
        lq_path = os.path.join(root, "lr_minmax")
        self.testset = GalaxyPairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        psnr, ssim = self.run(func, y_only, visualize, logging)
        print(f" {self.dataset:>8} - Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
        return psnr, ssim

    def run(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        crop_border = self.scale
        psnrs, ssims = [], []
        for i, (lq, gt) in enumerate(self.testset):
            sr = func(lq)
            sr = np.squeeze(sr)
            gt = np.squeeze(gt)
            # gt = gt * 255.
            # print(lq.dtype, sr.dtype, gt.dtype) sr == uint8
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)
            if logging:
                print(
                    f" {self.dataset:>8} - {i + 1:>3}/{len(self.testset):>3} PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}",
                    end="\r",
                )
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                compare([nn[:, :, ::-1], bc[:, :, ::-1], sr[:, :, ::-1], gt[:, :, ::-1]])
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        return psnr, ssim
    
class Test_Galaxy3264:
    def __init__(
        self,
        dataset: str = "Galaxy",
        scale: int = 2,
        root: str = "/data4/GalaxySynthesis/Galaxy_SR_Dataset/gt_64_lq_32/minmax/minmax_merge_ttv/test",
        save_path : str = "./"
    ) -> None:
        self.dataset = dataset
        self.scale = scale
        self.root = root
        self.save_path = save_path
        self.sr_save_path = save_path + '/sr'
        self.nn_save_path = save_path + '/nn'
        self.bc_save_path = save_path + '/bc'
        self.save_paths = [self.sr_save_path, self.nn_save_path, self.bc_save_path]
        gt_path = os.path.join(root, "gt_minmax")
        lq_path = os.path.join(root, "lr_minmax")
        self.scale = scale
        self.testset = Test_GalaxyPairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        psnr, ssim = self.run(func, y_only, visualize, logging)
        print(f" {self.dataset:>8} - Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
        return psnr, ssim

    def run(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        crop_border = self.scale
        psnrs, ssims = [], []

        if os.path.exists(self.save_path) == False:
            os.makedirs(self.save_path)
        if os.path.exists(self.sr_save_path) == False:
            os.makedirs(self.sr_save_path)
        if os.path.exists(self.nn_save_path) == False:
            os.makedirs(self.nn_save_path)
        if os.path.exists(self.bc_save_path) == False:
            os.makedirs(self.bc_save_path)

        for i, (file_name, lq, gt) in enumerate(self.testset):
            sr = func(lq)
            sr = np.squeeze(sr)
            gt = np.squeeze(gt)
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)

            if logging:
                print(
                    f" {self.dataset:>8} - {i + 1:>3}/{len(self.testset):>3} PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}",
                    end="\r",
                )            
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                lq = np.squeeze(lq)               
                file_name = file_name
                
                nn_psnr = compute_psnr(nn, gt, crop_border=crop_border, y_only=y_only)
                nn_ssim = compute_ssim(nn, gt, crop_border=crop_border, y_only=y_only)

                bc_psnr = compute_psnr(bc, gt, crop_border=crop_border, y_only=y_only)
                bc_ssim = compute_ssim(bc, gt, crop_border=crop_border, y_only=y_only)

                txt_file_path = os.path.join(self.save_path, "metrics.txt")

                # 텍스트 파일에 PSNR 및 SSIM 데이터 저장
                with open(txt_file_path, 'a') as f:
                    f.writelines(f'{file_name}.npy : SR > PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}\n')
                    f.writelines(f'{file_name}.npy : NN > PSNR: {nn_psnr:6.3f}, SSIM: {nn_ssim:6.4f}\n')
                    f.writelines(f'{file_name}.npy : BC > PSNR: {bc_psnr:6.3f}, SSIM: {bc_ssim:6.4f}\n')
                    f.writelines('---------------------------------------------------------------------\n')

                ours_compare([lq[:,::-1], nn[ :, ::-1], bc[ :, ::-1], sr[ :, ::-1], gt[ :, ::-1]], file_name = file_name, save_path=self.save_paths)
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        return psnr, ssim
    
class Test_Galaxy64128:
    def __init__(
        self,
        dataset: str = "Galaxy",
        scale: int = 2,
        root: str = "/data4/GalaxySynthesis/Galaxy_SR_Dataset/gt_128_lq_64/minmax/minmax_merge_ttv/test",
        save_path : str = "./"
    ) -> None:
        self.dataset = dataset
        self.scale = scale
        self.root = root
        self.save_path = save_path
        gt_path = os.path.join(root, "gt_minmax")
        lq_path = os.path.join(root, "lr_minmax")
        self.scale = scale
        self.testset = Test_GalaxyPairedImageDataset(gt_path, lq_path)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        psnr, ssim = self.run(func, y_only, visualize, logging)
        print(f" {self.dataset:>8} - Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}")
        return psnr, ssim

    def run(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        y_only: bool = True,
        visualize: bool = False,
        logging: bool = True,
    ) -> Tuple[float, float]:
        crop_border = self.scale
        psnrs, ssims = [], []
        for i, (file_name, lq, gt) in enumerate(self.testset):
            sr = func(lq)
            sr = np.squeeze(sr)
            gt = np.squeeze(gt)
            psnr = compute_psnr(sr, gt, crop_border=crop_border, y_only=y_only)
            ssim = compute_ssim(sr, gt, crop_border=crop_border, y_only=y_only)
            psnrs.append(psnr)
            ssims.append(ssim)
            save_path = self.save_path + f'/{file_name}'
            if logging:
                print(
                    f" {self.dataset:>8} - {i + 1:>3}/{len(self.testset):>3} PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}",
                    end="\r",
                )            
            if visualize:
                nn = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                bc = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                lq = np.squeeze(lq)               
                file_name = file_name
                
                nn_psnr = compute_psnr(nn, gt, crop_border=crop_border, y_only=y_only)
                nn_ssim = compute_ssim(nn, gt, crop_border=crop_border, y_only=y_only)

                bc_psnr = compute_psnr(bc, gt, crop_border=crop_border, y_only=y_only)
                bc_ssim = compute_ssim(bc, gt, crop_border=crop_border, y_only=y_only)

                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                txt_file_path = os.path.join(save_path, "metrics.txt")
                # 텍스트 파일에 PSNR 및 SSIM 데이터 저장
                with open(txt_file_path, 'w') as f:
                    f.writelines(f'SR > PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}\n')
                    f.writelines(f'NN > PSNR: {nn_psnr:6.3f}, SSIM: {nn_ssim:6.4f}\n')
                    f.writelines(f'BC > PSNR: {bc_psnr:6.3f}, SSIM: {bc_ssim:6.4f}\n')

                ours_compare([lq[:,::-1], nn[ :, ::-1], bc[ :, ::-1], sr[ :, ::-1], gt[ :, ::-1]], file_name = file_name, save_path=save_path)
            if i==10:
                break
        psnr = np.mean(psnrs)
        ssim = np.mean(ssims)
        return psnr, ssim