# Install with Containers

If you need to run simulations on a computing cluster, or the built-in installer doesn't work for you, pre-built Docker and Apptainer images with all dependencies are available on the [GitHub Container Registry](https://github.com/Harvard-Neutrino/prometheus/pkgs/container/prometheus).

If you're on a personal machine, go with Docker; if you're on a cluster, Apptainer is usually the better fit (and is often already installed).

!!! note
    The provided images are built for x86_64 and are not compatible with ARM-based architectures (e.g., Apple M-series Macs). If you are a Mac user on Apple Silicon, install from source instead.

## Available Images

| Tag           | Description       |
| ------------- | ----------------- |
| `:latest`     | CPU-only image    |
| `:latest-gpu` | GPU-enabled image |
| `:VERSION`    | Specific release  |

## Using Docker

If you need help getting started with Docker, see the [Docker documentation](https://docs.docker.com/desktop/). 

Here is an outline of the steps to set things up:

1. **Pull the image**

    ```sh
    docker pull ghcr.io/harvard-neutrino/prometheus:latest
    ```

2. **Run the container and launch a shell**

    ```sh
    docker run --rm -it ghcr.io/harvard-neutrino/prometheus:latest
    ```

3. **Run an example**

    ```sh
    docker run --rm -v "$PWD/output:/output" ghcr.io/harvard-neutrino/prometheus:latest python /opt/prometheus/examples/01_basic_water.py
    ```

    The `-v "$PWD/output:/output"` part of the command enables [volume mounting](https://docs.docker.com/engine/storage/volumes/#mounting-a-volume-over-existing-data) to access output files on your local machine.
    
For more container running options, refer to the [Docker containers documentation](https://docs.docker.com/engine/containers/run/).

### GPU Support

For GPU-accelerated simulations, you will need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Once installed, use the GPU image:

```sh
docker run --rm -it --gpus all ghcr.io/harvard-neutrino/prometheus:latest-gpu
```

If this fails, check that your NVIDIA drivers and container toolkit are correctly installed.

## Using Apptainer / Singularity

Apptainer (formerly Singularity) is particularly useful for running simulations on computing clusters, where it is often pre-installed. If you need help getting started, see the [Apptainer documentation](https://apptainer.org/docs/).

1. **Load the Apptainer module** (if on a cluster)

    ```sh
    module load apptainer
    ```

2. **Pull the container image**

    ```sh
    apptainer pull docker://ghcr.io/harvard-neutrino/prometheus:latest
    ```

3. **Run an example**

    ```sh
    apptainer exec prometheus_latest.sif python /opt/prometheus/examples/01_basic_water.py
    ```
