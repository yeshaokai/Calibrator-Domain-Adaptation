export PYTHONPATH=$PWD:$PYTHONPATH
#export LD_LIBRARY_PATH=/home/shaokai/miniconda3/envs/calibrator/lib/:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=1
#python -u scripts/train_calibrator.py --yaml_path configs/digits/svhn_to_mnist.yml #&> svhn_to_mnist.out 

#export CUDA_VISIBLE_DEVICES=1
#python -u scripts/train_calibrator.py --yaml_path configs/digits/usps_to_mnist.yml #&> usps_to_mnist.out &

export CUDA_VISIBLE_DEVICES=1
python -u scripts/train_calibrator.py --yaml_path configs/digits/mnist_to_usps.yml #&> mnist_to_usps.out

#export CUDA_VISIBLE_DEVICES=1
#python -u scripts/train_calibrator.py --yaml_path configs/digits/usps2mnist_to_mnist.yml &> usps2mnist_to_mnist.out&
#export CUDA_VISIBLE_DEVICES=2
#python -u scripts/train_calibrator.py --yaml_path configs/digits/mnist2usps_to_usps.yml &> mnist2usps_to_usps.out

#export CUDA_VISIBLE_DEVICES=3
#python -u scripts/train_calibrator.py --yaml_path configs/digits/gpu3.yml &> gpu3 &
#export CUDA_VISIBLE_DEVICES=4
#python -u scripts/train_calibrator.py --yaml_path configs/digits/gpu4.yml &> gpu4 &
#export CUDA_VISIBLE_DEVICES=5
#python -u scripts/train_calibrator.py --yaml_path configs/digits/gpu5.yml &> gpu5





