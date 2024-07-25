`~/.config/clangd/config.yaml`:
```yaml
CompileFlags:
  Add: [--cuda-gpu-arch=sm_86]
  Remove:
    - -rdc=true
    - -gencode
```
