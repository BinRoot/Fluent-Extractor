./svm_rank_learn -c 20 -t 1 -d 3 my_train.dat my_model
python mk_test_grid.py  # generates my_grid_test.dat
./svm_rank_classify my_grid_test.dat my_model my_grid_predictions
python visualize_predictions.py
