# Third Party

Source-only copies of the framework code used by the 0411 experiments:

```text
Mask2Former/
detectron2/
mmsegmentation/
```

Compiled artifacts are intentionally omitted from the submission tree. Rebuild
native extensions in the target environment when needed:

```bash
pip install -e third_party/detectron2
pip install -e third_party/mmsegmentation
cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
```
