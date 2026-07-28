# Completed SuperNI one-task TPU portability gate

This private Kaggle run is development-only, approximate TPU portability
evidence. It is not an exact paper reproduction, a paper-table cell, a
three-seed result, or evidence for the matched P100/A100 aggregate.

The run completed exactly one real SuperNI task with SLAO seed 42 from the
unchanged paper YAML. Training used one selected XLA device in TPU BF16 while
the runtime probe independently exposed all eight TPU v5 lite devices.
Autoregressive evaluation was offloaded to CPU float32 and is disclosed as
part of the portability implementation.

Finite loss, first-compile and post-first optimizer timing, total runtime, XLA
memory before and after training, and a CPU-loadable adapter checkpoint all
passed their remote validators. KJO cells, diagnostics-only download, the
strict run-directory audit, and the downloaded sensitive-artifact audit also
passed.

`summary.json` records the compact provenance, configuration hash, metrics,
timing, memory, checkpoint, audit results, and hashes. Detailed diagnostics
remain in the ignored local KJO run directory. The private remote kernel has
not been deleted because deletion was not authorized.
