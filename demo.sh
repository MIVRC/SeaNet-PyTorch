#SEAN for train
cd Train/

# SEAN x2  LR: 48 * 48  HR: 96 * 96
python main.py --template SEAN --save SEAN_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset

# SEAN x3  LR: 48 * 48  HR: 144 * 144
python main.py --template SEAN --save SEAN_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset

# SEAN x4  LR: 48 * 48  HR: 192 * 192
python main.py --template SEAN --save SEAN_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset




SEAN for test
cd Test/code/


#SEAN x2
python main.py --data_test MyImage --scale 2 --model SEAN --pre_train ../model/SEAN_x2.pt --test_only --save_results --chop --save "SEAN" --testpath ../LR/LRBI --testset Set5

#SEAN+ x2
python main.py --data_test MyImage --scale 2 --model SEAN --pre_train ../model/SEAN_x2.pt --test_only --save_results --chop --self_ensemble --save "SEAN_plus" --testpath ../LR/LRBI --testset Set5


#SEAN x3
python main.py --data_test MyImage --scale 3 --model SEAN --pre_train ../model/SEAN_x3.pt --test_only --save_results --chop --save "SEAN" --testpath ../LR/LRBI --testset Set5

#SEAN+ x3
python main.py --data_test MyImage --scale 3 --model SEAN --pre_train ../model/SEAN_x3.pt --test_only --save_results --chop --self_ensemble --save "SEAN_plus" --testpath ../LR/LRBI --testset Set5


#SEAN x4
python main.py --data_test MyImage --scale 4 --model SEAN --pre_train ../model/SEAN_x4.pt --test_only --save_results --chop --save "SEAN" --testpath ../LR/LRBI --testset Set5

#SEAN+ x4
python main.py --data_test MyImage --scale 4 --model SEAN --pre_train ../model/SEAN_x4.pt --test_only --save_results --chop --self_ensemble --save "SEAN_plus" --testpath ../LR/LRBI --testset Set5

