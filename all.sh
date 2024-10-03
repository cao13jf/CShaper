set -e

#python train.py --cf ConfigMemb/train_edt_discrete.txt --train_ratio 1
python test_edt.py --cf ConfigMemb/test.txt
