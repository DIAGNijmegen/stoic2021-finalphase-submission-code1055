# STOIC Challenge

## Prepare for Submission

1. Collect the checkpoints from the output directories of your runs that you want to use for the submission.
2. Wrap them up for submission: `python wrapup_checkpoint.py wrapped.pt ckp0.pt ckp1.pt ckp2.pt`
3. Put the generated checkpoint at the top level of this repository and name it `checkpoint.pt` (exact name is important).
4. Use the `container.sh` script to create builds for the submission using Docker:

``` sh
# to build the docker image
sudo ./container.sh build
# save the container image as a tar archive for submission
sudo ./container.sh save
# run the submission code and compare results
sudo ./container.sh validate
# open a shell inside the docker container
sudo ./container.sh shell
```

Running as `sudo` is important for Docker (or you are in the docker user group).

To use GPUs, you must install nvidia-container-toolkit: https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html

## Evaluate for Submission

Use `submission_inference.py` to predict multiple patients using the current docker configuration. Follow the steps mentioned above to prepare the checkpoint. Since it runs docker, you probably need to use sudo to run `submission_inference.py`.

## Build Dataset Caches

Run the `build_cache.py` script in the misc_utilities folder to build the cache for the current configuration:

    python -m misc_utilities.build_cache
