docker pull nvcr.io/nvidia/tritonserver:24.02-py3-igpu

docker run \
  --restart=always \
  --net=host \
  -v /home/vanh/anhnv/color_paper/models/triton:/models \
  --name color_paper \
  -d \
  --runtime nvidia \
  nvcr.io/nvidia/tritonserver:24.02-py3-igpu \
  tritonserver --model-repository=/models