# Context-Awareness and Multi-Branch Attention Fusion for Visible-Infrared Person Re-Identification
![](assets/hmcl3.jpg)
A  project for visible-infrared person re-identification, featuring a context-aware multi-branch attention fusion framework – the code implementation and supplement to the related research.
## Contributions
1. We designed a Hierarchical Context Extractor (HCE) that employs dilated convolutions and residual structures to model relationships between human body parts using lightweight parameters. It collaborates with ResNet to extract local details, constructing a comprehensive cross-modal identity representation while effectively mitigating modal discrepancies.
2. We constructed a Multi-Branch Unified Encoder (MBUE) that achieves initial decoupling of modal features via a parallel branch structure. This not only enhances the diversity of feature representations but also effectively addresses the interference of modal discrepancies on feature alignment.
3. We propose the Channel-Spatial-Cross Attention-Guided Cross-Modality Alignment (CSC-CMA) module.It enhances discriminative features and fully separates modal features via channel and spatial attention, then achieves precise cross-modal alignment through cross-attention and adaptive feature interaction.
