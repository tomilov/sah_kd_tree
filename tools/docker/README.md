 > pamac install nvidia-container-runtime
or
 > apt-get install nvidia-container-runtime
add to /etc/docker/daemon.json:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         }
    },
    "default-runtime": "nvidia"
}
```
then
 > systemctl restart docker

