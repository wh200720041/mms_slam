#WEIGHTS=work_dirs/cityscapes/solov2_release_r50_fpn_2gpu_3x/latest.pth
WEIGHTS=$1
python3 tools/test_ins.py configs/solov2/cityscapes/solov2_r50_fpn_2gpu_3x.py $WEIGHTS --show --out  results_solo.pkl --json_out results_solo_cityscapes --eval segm
