export AWS_DEFAULT_REGION=us-east-1
REGION=us-east-1
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin 684288478426.dkr.ecr.${REGION}.amazonaws.com

random_tag=$(printf '%s' $(echo "$RANDOM" | md5sum) | cut -c 1-24)
DOCKER_IMAGE_TAG=nemo-eval
sudo docker build --tag=$DOCKER_IMAGE_TAG:${random_tag} -f Dockerfile_verl.local .
echo "Built docker image nemo-eval:${random_tag}"
echo "Running docker image nemo-eval:${random_tag}"

sudo docker run --gpus all -p 8011:8011 \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --name=$DOCKER_IMAGE_TAG-${random_tag} \
    -v /ebs-basemodeling/:/ebs-basemodeling \
    --mount type=tmpfs,destination=/tmpfs $DOCKER_IMAGE_TAG:${random_tag}