# LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models
[Long Lian](https://tonylian.com/), [Baifeng Shi](https://bfshi.github.io/), [Adam Yala](https://www.adamyala.org/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Boyi Li](https://sites.google.com/site/boyilics/home) at UC Berkeley/UCSF.

[Paper](https://arxiv.org/abs/2309.17444) | [Project Page](https://llm-grounded-video-diffusion.github.io/) | [HuggingFace Demo (coming soon)](#) | [Related Project: LMD](https://llm-grounded-diffusion.github.io/) | [Citation](#citation)

![Comparisons with our baseline](https://llm-grounded-video-diffusion.github.io/teaser.jpg)

![Method Figure](https://llm-grounded-video-diffusion.github.io/overall_method.jpg)

Our DSL-grounded Video Generator:

![DSL-grounded Video Generator](https://llm-grounded-video-diffusion.github.io/dsl_to_video.jpg)

LLM generates dynamic scene layouts, taking the world properties (e.g., gravity, elasticity, air friction) into account:

![](https://llm-grounded-video-diffusion.github.io/world_properties.jpg)

LLM generates dynamic scene layouts, taking the camera properties (e.g., perspective projection) into account:

![](https://llm-grounded-video-diffusion.github.io/camera_properties.jpg)

We propose a benchmark of five tasks. Our method improves on all five tasks without specifically aiming for each one:

![](https://llm-grounded-video-diffusion.github.io/visualizations.jpg)

## Code
The code is coming soon! Meanwhile, give this repo a star to support us!

## Contact us
Please contact Long (Tony) Lian if you have any questions: `longlian@berkeley.edu`.

## Citation
If you use our work or our implementation in this repo, or find them helpful, please consider giving a citation.
```
@article{lian2023llmgrounded,
      title={LLM-grounded Video Diffusion Models}, 
      author={Lian, Long and Shi, Baifeng and Yala, Adam and Darrell, Trevor and Li, Boyi},
      journal={arXiv preprint arXiv:2309.17444},
      year={2023},
}
```
