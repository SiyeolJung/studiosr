import os
from studiosr import Test_Galaxy3264, Trainer
from studiosr.models import HAT
from studiosr.utils import get_device
from studiosr.data import Galaxy

scale = 2  # 2, 3, 4
dataset = "Galaxy"  # Set5, Set14, BSD100, Urban100, Manga109
n_colors = 1
device = get_device()
model_dir = ".pth file path"  # '/home3/t202401082/studiosr/checkpoints_hat_3264_float_fixed_12_minmax_floatformat_excludezerodata_lr00002'
test_dir = "test folder path" # "/data4/GalaxySynthesis/Galaxy_SR_Dataset/gt_64_lq_32/minmax/minmax_merge_ttv/test"
save_path =  "saved folder path"  #'/home3/t202401082/studiosr/results_hat_3264_float_fixed_12_minmax_floatforamt_excludezerodata_lr00002'
if os.path.exists(model_dir) == False:
    os.makedirs(model_dir)
model = HAT.from_our_pretrained(scale=scale,n_colors=n_colors,model_dir=model_dir).eval().to(device)
evaluator = Test_Galaxy3264(dataset, scale=scale, root=test_dir, save_path=save_path)
psnr, ssim = evaluator(model.inference, visualize=True)

txt_file_path = os.path.join(save_path, 'result.txt')
with open(txt_file_path, 'w') as f:
    f.writelines(f'Average PSNR: {psnr:6.3f}, SSIM: {ssim:6.4f}')