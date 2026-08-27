# Unified ResNet Relax/MetaSchedule Results

MetaSchedule plus DLight schedule coverage: pass.
Every workload has a measured MetaSchedule record: False.
Paired execution check: pass.

## MetaSchedule Coverage

| Model | MS measured | DLight fallback | Records | Failed | Sum best ms |
|---|---:|---:|---:|---:|---:|
| resnet18 | 15/19 | 4 | 1029 | 661 | 0.690522 |
| resnet34 | 13/19 | 6 | 1029 | 609 | 0.455717 |
| resnet50 | 26/28 | 2 | 1030 | 310 | 1.274343 |

## Paired Default-Pipeline Execution

| Model | Batch | CUDA p50 ms | SYCL p50 ms | Ratio | Pass |
|---|---:|---:|---:|---:|:---:|
| resnet18 | 1 | 2.502704 | 4.400721 | 1.758x | True |
| resnet18 | 8 | 3.613291 | 6.799239 | 1.882x | True |
| resnet18 | 16 | 5.540703 | 10.695794 | 1.930x | True |
| resnet34 | 1 | 5.170603 | 8.531096 | 1.650x | True |
| resnet34 | 8 | 6.583657 | 13.248483 | 2.012x | True |
| resnet34 | 16 | 10.385424 | 20.980260 | 2.020x | True |
| resnet50 | 1 | 3.920989 | 7.425904 | 1.894x | True |
| resnet50 | 8 | 6.581522 | 13.901580 | 2.112x | True |
| resnet50 | 16 | 9.902766 | 23.362647 | 2.359x | True |
