# DATA — excluded inputs

No training text, rank caches, corpus identifiers, embeddings, fitted weights or checkpoints ship. Even small arrays are excluded when fitted on private inputs. The following inventory records local source bytes, not a download offer. Private inputs are not available for redistribution; substitute your own licensed corpus.

| Relative source file | Bytes | SHA-256 |
|---|---:|---|
| `curriculum/levels/L0.jsonl` | 1183648 | `030e5588a50f5b6d9eeb0e9b26013f20e1fa0ab23180545fef5d6fbc91da5465` |
| `curriculum/levels/L0.nomic.npy` | 1843328 | `2138ccbe88a90d8099e22b591d64e80a9726c04801060d3c3a7f2a9f4377c46a` |
| `curriculum/levels/L1.jsonl` | 3731438 | `39226ac8832ce08d1661d5459972f9189019a8ac746e7944d4844b36d0ff35cd` |
| `curriculum/levels/L2.jsonl` | 215536 | `51669fc54047be111468983e1bc9829692bf94825b70b73043f1a021ffb4eae8` |
| `curriculum/levels/L3.jsonl` | 780422 | `bef5bae71f502d7edf2f50ffc1d02d46361c4e03869b92553fbe40a962d9306d` |
| `curriculum/levels/L4.jsonl` | 2823935 | `1b36a667928d75271391c67680eebb74b95c86d64c661e80ee1c619675abaa23` |
| `curriculum/prek/probes.jsonl` | 137718 | `04957f1f779288b0caa2c72a1edadfb49aa730cf8af0e756a70ea2d4e3ca0c19` |
| `e1/pca256.npy` | 789632 | `46c356eb8afdcaff146f25a0e54fdfc0988da4afa010ca3ad40bd814068f93f1` |
| `e1/sif_weights.npy` | 122216 | `294ab3680eafaf74fabedf7ec5728acbb9e06c473bbde82e17f17c834cd2edff` |
| `e1/token_table_f32.npy` | 93763712 | `1f7d82f452c1bbca4b6228950e6bd140e1171b0ae44cff71cdd68e4d0738e8c7` |
| `e1/vocab/part.000.emb` | 16521919 | `32c491c0e7f69039c4e137944ecba5e7a00458ed4cda08afca3d74db8c56d7b4` |
| `e1/vocab/part.001.emb` | 16520479 | `bceebd78dd5728bcfc5c642b3b8c068ea82a4ea3c52060198a16b2790d5f8430` |
| `e1/vocab/part.002.emb` | 16519611 | `e2ae8e06c3c39dfced7c42052c9f65aa210a4b8cb7ee36a85a4ec98a3c21c262` |
| `e1/vocab/part.003.emb` | 16519516 | `03f62dce159cbcbb9e68b58ec5d1a29ea69630840fce3ddba443735f7ba42159` |
| `e1/vocab/part.004.emb` | 16519147 | `95ff783982dcaa2e3562caef228592e11c8b373e333c319c257633c805eccea4` |
| `e1/vocab/part.005.emb` | 16519394 | `38fd17924989cddc39b900c14d0a3a3727e1298603033ea2a09ea2f9dd567d98` |
| `e1/vocab/part.006.emb` | 16519534 | `0ef1ff8858377cb0118bccbc99919441ba40261aafa4998a8de61db2232bdea4` |
| `e1/vocab/part.007.emb` | 16519650 | `d2d62f4915d173ccb143e09b8f83abfe65bf3ab2381267c039002794b5d0b99a` |
| `e1/vocab/part.008.emb` | 16520029 | `850ae8aa3140f23b787cd84fac9df8443f386486a5e5a731307460dc8a099820` |
| `e1/vocab/part.009.emb` | 16519526 | `1421a9ffcf3601b9a2e663a0ecc1046d81e5c87ec9ffb09c0e58b75c12042faf` |
| `e1/vocab/part.010.emb` | 16519570 | `108b794ca1026f0021c355866857eacb8ccb31ce01333ef1a126dce587ba6092` |
| `e1/vocab/part.011.emb` | 16518966 | `3fae9ebd064de1f8ac495ab1cceae2d330dd01928adfd10afd67ba3822dd8e29` |
| `e1/vocab/part.012.emb` | 16519452 | `dcd1639b8e32eee69342bc5affce074ab107a4df6034205fbbf2c28ffe5431f6` |
| `e1/vocab/part.013.emb` | 16519322 | `bf29e305dd1499c2718e6d860e211751dec227885dd44360ec41efd05d5ef709` |
| `e1/vocab/part.014.emb` | 14921113 | `32addff58cdf736bdf56457cd9f91d608e2c658aef5dd42aa1c0d916097c379b` |
| `e5/token_table_learned_l0.0001_s150.npy` | 46881920 | `ab8082e9cd9b468361e76047e49c8f870047bddd1607fd1e20a9424df2d7f1f3` |
| `e6/fitset.npz` | 74962823 | `0c5c7027b1cc4a2d29d59aefe62e947ccbeeb793706b778f5b03ad2efcbbe967` |
| `e6/table_l0.0001_s300.npy` | 46881920 | `9af694202af0d595bb287516f0db797802f9c0f90b5dd0698be77322e1f04f39` |
| `e8/Wo_ordered.npy` | 19664000 | `aa707dcb6bf6b11769335ffe260d4c3615320bfd9bc0dfa4a297aa0a36ac63f7` |
| `e9/e9_reservoir.pt` | 56168144 | `4628413e8c305248da0e490a6480bd04b9436f8b31475e8fc7f809c7e2aa336c` |
| `results/heldout-1013/rank-e9.json` | 1626841 | `c421128b438bc390576aec6cde0f852df27e6a44dbd63de719d2827bd2421a32` |

## Rebuilding with your own inputs
Place your separately licensed nomic teacher, vocabulary and tokenizer under `~/projects/bitnet/` as named in `e1/distil.sh`, then run `bash e1/distil.sh`. `python3 e4/eval.py` concatenates the per-token teacher outputs and saves `e1/token_table_f32.npy`, `e1/sif_weights.npy`, and `e1/pca256.npy`; it also requires your own document index and shardd-compatible store. The shipped hashes identify the historical files, not reproducibility on a different corpus. `e6/prep.py` and `e6/fit.py` prepare and fit user-owned text; fit.py expects fitset.npz and token_table_f32.npy in its working directory. E7 needs the external GGUF/BitNet stack, E8 needs core-model weights, E9 needs PyTorch and a fitted checkpoint; T1 additionally needs the external T50 evaluator and omni-ring. Curriculum build/eval/selftest need user-owned levels and weights; only the pure gate logic runs here.

## CLI paths
`bin/embed-core` reads optional `EMBED_CORE_WEIGHTS`, `EMBED_CORE_TOKENIZER`, `EMBED_CORE_TABLE`, `EMBED_CORE_OMNID`, `EMBED_CORE_INDEX` and `EMBED_CORE_ARMILLA` environment variables. Defaults use sibling core-models/weights, local data/tokenizer.json, e1/token_table_f32.npy, sibling omnid, local data/index and sibling liquid-armilla. The other historical experiment scripts use the relative `<your-data>` placeholder paths visible in their source; supply your own inputs and adjust those paths before running external experiments. No private deployment is contacted by a quick check.
