# Data

Raw datasets are not stored in this repository. Keep large datasets outside Git and pass paths through args or environment variables.

Common args:

```text
--test-root
--cosec-root
--brenet-root
--cosec-manifest
--dsec-root
--acdc-root
```

Common environment variables:

```bash
export COSEC_ROOT=/path/to/cosec/train
export TEST_ROOT=/path/to/cosec/test
export BRENET_ROOT=/path/to/BRENet
export EVENTSHIFT_COSEC_MANIFEST=/path/to/cosec_train_bidir_50ms.json
export DSEC_ROOT=/path/to/dsec
export DSEC_FILTERED_630_MANIFEST=/path/to/dsec19_filtered_medium_more_630.json
export ACDC_ROOT=/path/to/acdc
export ACDC_SPLIT_DIR=/path/to/acdc/splits
export EVENTSHIFT_COSEC_SPLIT_DIR=/path/to/cosec/splits
```

See `../docs/dataset_preparation.md` for expected layouts, CoSEC k-fold splits, frame-list prefix splits, and domain-gap filtering.
