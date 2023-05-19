# IBM OPC TEST source optimized dataset

## File tree

```text
.
├── annular_RI
├── annular_SO
├── annular_mask
├── dipole_RI
├── dipole_SO
├── dipole_mask
├── ibm_opc_test.md
├── quasar_RI
├── quasar_SO
└── quasar_mask
```

### 光源名字

按照之前的三元组：我分了三个文件夹 `RI`, `SO` 和 `mask` , 每个文件都有前缀，分别是`annular`, `dipole`, `quasar`，他们分别代表的是 `环形光源`，`双扇形光源`, `四扇形光源` 。我这里并没有把光源的初始形状给你，如果需要的话我也可以发给你。

### 文件名字

- `SO` 这个是经过优化后的mask，目前是 $51 \\times 51$ 的矩阵，这个大小你可以任意降采样或者升采样。只要符合你网络结构就可以。
- `mask` 就是mask。所有的`annular_mask`, `dipole_mask`, `quasar_mask`内容都是一样的。我是额外存了。
- `RI` 这个是经过光刻以后形成 的图案。在你的网络里面应该是用不上的，不过我一并存了。
