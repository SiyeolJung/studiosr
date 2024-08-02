from studiosr import Evaluator_Galaxy3264, Trainer
from studiosr.data import Galaxy
from studiosr.models import HAT

dataset_dir= "Dataset dir" # "/data4/GalaxySynthesis/Galaxy_SR_Dataset/gt_64_lq_32/minmax/minmax_merge_ttv"
ckpt_path = "ckpt save path"#"checkpoints_hat_3264_float_fixed_12_minmax_floatformat_excludezerodata_lr00002"
scale = 2
size = 32
n_colors = 1
batch_size = 32
learning_rate = 0.0002
dataset = Galaxy(
    dataset_dir=dataset_dir,
    scale=scale,
    size=size,
    transform=False, # data augmentations
    to_tensor=True,
    download=False, # if you don't have the dataset
)
evaluator = Evaluator_Galaxy3264(scale=scale)

model = HAT(scale=scale,n_colors=n_colors)
trainer = Trainer(model, dataset, evaluator, batch_size=batch_size, ckpt_path=ckpt_path, eval_interval=1000, learning_rate=learning_rate)
trainer.run()