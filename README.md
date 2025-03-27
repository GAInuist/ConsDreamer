# ConsDreamer: Advancing Multi-View Consistency for Zero-Shot Text-to-3D Generation



<div align="center">
  <img src="resources/results.png"width="80%" />
</div>

## Abstract
We propose a novel approach, *ConsDreamer*, which generates 3D content with multi-view consistency while effectively mitigating the multi-face Janus problem.
<details>
<summary>Click for the full abstract <b></b></summary>
Recent advances in zero-shot text-to-3D generation have revolutionized 3D content creation by enabling direct synthesis from textual descriptions. While state-of-the-art (SOTA) methods leverage 3D Gaussian Splatting with score distillation to enhance multi-view rendering through pre-trained T2I models, they suffer from inherent view biases in T2I priors that lead to inconsistent 3D generation, particularly manifesting as the multi-face Janus problem, where objects exhibit conflicting features across views.

To address this fundamental challenge, we propose **ConsDreamer**, a novel framework that mitigates view bias by refining both the conditional and unconditional terms in the score distillation process:
- **View Disentanglement Module (VDM)**: Eliminates viewpoint biases in conditional prompts by decoupling irrelevant view components and injecting precise camera parameters
- **Similarity-based partial order loss**: Enforces geometric consistency in the unconditional term by aligning cosine similarities with azimuth relationships

Extensive experiments validate that ConsDreamer effectively mitigates the multi-face Janus problem in text-to-3D generation, surpassing existing methods in both quality and consistency.
</details>

## Pipeline
<div align="center">
  <img src="resources/total_pipeline.png" width="60%" />
  <img src="resources/pipeline2.png" width="35%" /> 
</div>




