`~/.config/clangd/config.yaml`:
```yaml
CompileFlags:
  Add:
    - --cuda-gpu-arch=sm_75
    - --cuda-path=/opt/cuda
    - -I/opt/cuda/include
    - --cuda-host-only
  Remove:
    - -rdc=true
    - -gencode
```
