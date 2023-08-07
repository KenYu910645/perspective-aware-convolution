g++ -O3 -DNDEBUG -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp

# ./evaluate_object_3d_offline \
# /home/lab530/KenYu/3d-bounding-box-estimation-for-autonomous-driving/training/label_2/ \
# /home/lab530/KenYu/3d-bounding-box-estimation-for-autonomous-driving/training/box_3d/ \
# /home/lab530/KenYu/ml_toolkit/kitti/devkit/cpp/viz/ 

# Debug
# ./evaluate_object_3d_offline \
# /home/lab530/KenYu/3d-bounding-box-estimation-for-autonomous-driving/bug/label_2/ \
# /home/lab530/KenYu/3d-bounding-box-estimation-for-autonomous-driving/bug/box_3d/ \
# /home/lab530/KenYu/ml_toolkit/kitti/devkit/cpp/viz/ 

# DD3D
./evaluate_object_3d_offline \
/data/datasets/KITTI3D/training/label_2/ \
/home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions_standard_format/ \
/home/lab530/KenYu/ml_toolkit/kitti/devkit/cpp/dd3d_viz/ > result.txt; cat result.txt


