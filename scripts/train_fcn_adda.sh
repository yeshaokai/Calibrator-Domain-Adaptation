export PYTHONPATH=/home/shaokai/data_calibrator:$PYTHONPATH
gpu=1,2,3,4,5,6

######################
# loss weight params #
######################
lr=1e-6
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=10001
crop=768
snapshot=5000
batch=6

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='cyclegta5'
tgt='cityscapes'
datadir='/home/lthpc/gta2cityscape/'


resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
model='drn26'
baseiter=115000
#model='fcn8s'
#baseiter=100000


base_model="base_model/${model}-${src}-iter${baseiter}.pth"
outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

# Run python script #
CUDA_VISIBLE_DEVICES=${gpu} python -u scripts/train_fcn_calibrator.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu ${gpu} \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot
