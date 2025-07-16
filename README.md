# Context-Awareness and Multi-Branch Attention Fusion for Visible-Infrared Person Re-Identification
![](assets/hmcl3.jpg)
A  project for visible-infrared person re-identification, featuring a context-aware multi-branch attention fusion framework – the code implementation and supplement to the related research.
## Contributions
1. We designed a Hierarchical Context Extractor (HCE) that employs dilated convolutions and residual structures to model relationships between human body parts using lightweight parameters. It collaborates with ResNet to extract local details, constructing a comprehensive cross-modal identity representation while effectively mitigating modal discrepancies.
2. We constructed a Multi-Branch Unified Encoder (MBUE) that achieves initial decoupling of modal features via a parallel branch structure. This not only enhances the diversity of feature representations but also effectively addresses the interference of modal discrepancies on feature alignment.
3. We propose the Channel-Spatial-Cross Attention-Guided Cross-Modality Alignment (CSC-CMA) module.It enhances discriminative features and fully separates modal features via channel and spatial attention, then achieves precise cross-modal alignment through cross-attention and adaptive feature interaction.

## Results

\usepackage{booktabs} % 引入三线表宏包
\begin{table*}[htbp]
\centering
\caption{Comparison of the CMFN Method with State-of-the-Art Methods on the SYSU-MM01 Dataset.}
\label{tab:SYSU}
% 调整表格与上下文间距
\resizebox{1\textwidth}{!}{% 自动调整表格宽度适应页面，可根据需要调整或删除
\begin{tabular}{l c c c c c c c c c c c}
\toprule
\multirow{2}{*}{Methods} & \multirow{2}{*}{Venue} & \multicolumn{5}{c}{All Search} & \multicolumn{5}{c}{Indoor Search} \\
\cmidrule(lr){3-7} \cmidrule(lr){8-12}
& & Rank-1 & Rank-10 & Rank-20 & mAP & mINP & Rank-1 & Rank-10 & Rank-20 & mAP & mINP \\
\midrule
AGW\cite{Ye2021Deep} & TPAMI-21 & 47.5 & - & - & 47.6 & 35.3 & 54.1 & - & - & 62.9 & 59.2 \\
NFS\cite{Chen2021Neural} & CVPR-21 & 56.9 & 91.3 & 96.5 & 55.4 & - & 62.7 & 96.5 & 99.0 & 69.7 & - \\
MPANet\cite{Wu2021Discover} & CVPR-21 & 70.5 & 96.2 & 98.8 & 68.2 & - & 76.7 & 98.2 & 99.5 & 80.9 & - \\
MCLNet\cite{Hao2021CrossModality} & ICCV-21 & 65.4 & 93.3 & 97.1 & 61.9 & 47.3 & 72.5 & 96.9 & 99.2 & 76.5 & 72.1 \\
SMCL\cite{Wei2021Syncretic} & ICCV-21 & 67.3 & 92.8 & 96.7 & 61.7 & - & 68.8 & 96.5 & 95.2 & 98.7 & 75.5 \\
FMCNet\cite{Zhang2022FMCNet} & CVPR-22 & 66.3 & - & - & 62.5 & - & 68.1 & - & - & 74.0 & - \\
DART\cite{Yang2022Learning}  & CVPR-22 & 68.7 & 96.3 & 98.9 & 66.2 & 53.2 & 72.5 & 97.8 & 99.4 & 78.1 & 74.9 \\
MAUM$^P$\cite{Liu2022Learning} & CVPR-22 & 71.6 & - & - & 68.7 & - & 76.9 & - & - & 81.9 & - \\
PMT\cite{Lu2023Learning} & AAAI-23 & 67.5 & 95.3 & 95.3 & 64.9 & 51.8 & 71.6 & 96.7 & 99.2 & 76.5 & 72.7 \\
MRCN\cite{Zhang2023MRCN} & AAAI-23 & 68.9 & 95.2 & 98.4 & 65.5 & - & 76.0 & 98.3 & 99.7 & 79.8 & - \\
DEEN\cite{Zhang2023Diverse} & CVPR-23 & 74.7 & 97.6 & 99.2 & 71.6 & - & 80.3 & 99.0 & 99.8 & 83.3 & - \\
SEFEL\cite{Feng2023ShapeErased} & CVPR-23 & 75.2 & 96.9 & 99.1 & 70.1 & - & 78.4 & 97.5 & 98.9 & 81.2 & - \\
PartMix\cite{Kim2023Partmix} & CVPR-23 & 77.8 & - & - & 74.6 & - & 81.5 & - & - & 84.4 & - \\
IDKL$^\text{a}$\cite{Ren2024ImplicitDiscriminative} & CVPR-24 & 70.8 & 95.6 & 98.3 & 68.5 & - & 79.9 & 97.7 & 99.3 & 82.7 & - \\
CAJ\cite{Ye2024ChannelAugmentation} & TPAMI-24 & 71.4 & 96.2 & 98.7 & 68.1 & 54.3 & 78.3 & 98.3 & 99.7 & 81.9 & \underline{78.4} \\
TMD\cite{Lu2024TriLevel} & TMM-24 & 73.9 & 96.2 & 98.7 & 67.7 & 51.5 & 81.1 & 98.8 & 99.6 & 78.8 & 73.3 \\
DCPLNet\cite{Chan2024DiverseFeature} & TII-24 & 74.0 & 96.5 & 98.9 & 70.4 & - & 78.3 & 98.7 & 99.8 & 81.9 & - \\
HOS-Net\cite{Qiu2024High} & AAAI-24 & 75.6 & - & - & 74.2 & - & 84.2 & - & - & 86.7 & - \\
MHN\cite{Zuo2025Modality} & IF-25 & 75.0 & 97.2 & - & 71.8 & - & 81.5 & 98.8 & - & 85.0 & - \\
SCR\cite{Yu2025No} & IF-25 & 75.5 & 97.7 & \underline{99.4} & 71.2 & - & 84.7 & 99.1 & \underline{99.9} & 86.1 & - \\
AMML\cite{Zhang2025Adaptive} & IJCV-25 & 77.8 & \underline{97.8} & \underline{99.4} & 74.8 & - & \underline{86.6} & \textbf{99.5} & \textbf{100.0} & \underline{88.3} & - \\
MSCMNet\cite{HUA2025111090} & PR-25 & 78.5 & 97.5 & 99.2 & 74.2 & - & 83.0 & 98.9 & 99.8 & 85.5 & - \\
DEEN$+$DiVE\cite{Dai2025Diffusion} & AAAI-25 & \underline{79.0} & - & - & \underline{74.9} & - & 82.9 & - & - & 85.9 & - \\
\midrule
CMFN(Ours) & - & \textbf{79.5} & \textbf{98.5} & \textbf{99.6} & \textbf{76.4} & \textbf{64.7} & \textbf{87.6} & \underline{99.3} & \underline{99.9} & \textbf{89.0} & \textbf{86.0} \\
\bottomrule
\end{tabular}
}
\end{table*}

## Datasets


## Environment

## Training

## Evaluation
